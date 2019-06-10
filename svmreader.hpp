#pragma once
#include <uv.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <boost/bind.hpp>
#include <boost/atomic.hpp>
#include <boost/thread/thread.hpp>
#include <boost/lockfree/spsc_queue.hpp>

#include "data.hpp"

#define printf

using namespace boost;

class svm_reader {
    public:
        example_data& storage;
        const std::string input_file;
        svm_reader(const std::string input_file, example_data& storage) : input_file(input_file), storage(storage) {
            // BOOST_LOG_TRIVIAL(info) << "libsvm file reader initialized with input: " << input_file;
        }
        /*!brief load libsvm format input file */
        bool load() {
            /* guarantees that all worker threads are on the same NUMA node */
            /* initialize temp buffers and state variables */
            parse_buf.reserve(64);
            parse_buf.clear();
            state = STATE_NEW;
            last_is_blank = true;
            cur_example_id = -1;
            cur_feature_id = -1; // no features read yet
            line_no = 0;
            /* prepare feature buffer */
            item_buf_index = 0;
            item_index = 0;
            item_buf[0] = new std::tuple<i_type, i_type, d_type>[ItemBufferSize];
            item_buf[1] = new std::tuple<i_type, i_type, d_type>[ItemBufferSize];
            work_done = true;
            work_ready = false;
            exit_worker = false;
            boost::thread* feat_worker = new boost::thread(boost::bind(&svm_reader::feat_store_proc, this));
            /* open file */
            fd = uv_fs_open(uv_default_loop(), &open_req, input_file.c_str(), O_RDONLY, 0, NULL);
            if (fd == -1) {
                std::cerr << "Cannot open input file " << input_file;
                return false;
            }
            buffers[0] = (char *)malloc(IOBufferSize);
            buffers[1] = (char *)malloc(IOBufferSize);
            if (buffers[0] == NULL || buffers[1] == NULL) {
                std::cerr << "Cannot allocate buffer memory " << input_file;
                return false;
            }
            iov_index = 0;
            iovs[0] = uv_buf_init(buffers[0], IOBufferSize);
            iovs[1] = uv_buf_init(buffers[1], IOBufferSize);
            read_req.data = this;
            uv_fs_read(uv_default_loop(), &read_req, fd, &iovs[0], 1, -1, svm_reader::on_read);
            uv_run(uv_default_loop(), UV_RUN_DEFAULT);
            uv_fs_req_cleanup(&open_req);
            uv_fs_req_cleanup(&read_req);
            free(buffers[0]);
            free(buffers[1]);
            /* cleanup worker */
            exit_worker = true;
            {
                boost::lock_guard<boost::mutex> lock(work_ready_mux);
                work_ready = true;
            }
            work_ready_cond.notify_one();
            feat_worker->join();
            for (size_t i = 0; i < item_index; ++i) {
                std::tuple<i_type, i_type, d_type>& p = item_buf[item_buf_index][i];
                storage.add_feature_val(std::get<0>(p), std::get<1>(p), std::get<2>(p));
            }
            delete [] item_buf[0];
            delete [] item_buf[1];
            delete feat_worker;
            return true;
        }
    private:
        enum paser_states {
            STATE_NEW, STATE_FEAT, STATE_VAL, STATE_BLANK
        };
        /* variables for libuv */
        /*!brief read buffer size*/
        static const size_t IOBufferSize = 8192 * 1024;
        int fd;
        uv_fs_t open_req;
        uv_fs_t read_req;
        /*!brief double-buffered libuv file IO */
        uv_buf_t iovs[2];
        int iov_index;
        char* buffers[2];
        /* variables for the state machine */
        paser_states state;
        int64_t cur_example_id;
        double cur_y_value;
        int64_t cur_feature_id;
        double cur_feature_value;
        std::string parse_buf;
        bool last_is_blank;
        size_t line_no;
        /* variables for item buffering and storing */
        static const size_t ItemBufferSize = 16384;
        boost::condition_variable work_done_cond;
        boost::mutex work_done_mux;
        bool work_done;
        boost::condition_variable work_ready_cond;
        boost::mutex work_ready_mux;
        bool work_ready;
        boost::atomic<bool> exit_worker;
        int item_buf_index;
        int item_index;
        std::tuple<i_type, i_type, d_type>* item_buf[2];
        

        inline bool isblank(char c) {
            return (c == '\t' || c == ' ');
        }

        /*!brief A fast atoi implementation, works for unsigned numbers (feature IDs) only! */
        inline uint64_t positive_atoi(const char * str ) {
            uint64_t val = 0;
            while(isdigit(*str)) {
                val = val * 10 + (*str++ - '0');
            }
            return val;
        }


        /*!
         * \brief A faster version of strtof
         * TODO the current version does not support INF, NAN, and hex number
         */
        float strtof(const char *nptr) {
            const char *p = nptr;
            // Skip leading white space, if any. Not necessary
            while (*p <= ' ') ++p;

            // Get sign, if any.
            bool sign = true;
            if (*p == '-') {
                sign = false; ++p;
            } else if (*p == '+') {
                ++p;
            }

            // Get digits before decimal point or exponent, if any.
            float value;
            for (value = 0; isdigit(*p); ++p) {
                value = value * 10.0f + (*p - '0');
            }

            // Get digits after decimal point, if any.
            if (*p == '.') {
                uint64_t pow10 = 1;
                uint64_t val2 = 0;
                ++p;
                while (isdigit(*p)) {
                    val2 = val2 * 10 + (*p - '0');
                    pow10 *= 10;
                    ++p;
                }
                value += static_cast<float>(
                        static_cast<double>(val2) / static_cast<double>(pow10));
            }

            // Handle exponent, if any.
            if ((*p == 'e') || (*p == 'E')) {
                ++p;
                bool frac = false;
                float scale = 1.0;
                unsigned expon;
                // Get sign of exponent, if any.
                if (*p == '-') {
                    frac = true;
                    ++p;
                } else if (*p == '+') {
                    ++p;
                }
                // Get digits of exponent, if any.
                for (expon = 0; isdigit(*p); p += 1) {
                    expon = expon * 10 + (*p - '0');
                }
                if (expon > 38) expon = 38;
                // Calculate scaling factor.
                while (expon >=  8) { scale *= 1E8;  expon -=  8; }
                while (expon >   0) { scale *= 10.0; expon -=  1; }
                // Return signed and scaled floating point result.
                value = frac ? (value / scale) : (value * scale);
            }

            return sign ? value : - value;
        }

        /*!brief thread for pushing new pairs*/
        void feat_store_proc () {
            long long sum = 0;
            while(true) {
                {
                    boost::unique_lock<boost::mutex> lock(work_ready_mux);
                    //puts("3");
                    while (!work_ready) {
                        work_ready_cond.wait(lock);
                    }
                    work_ready = false;
                }
                if (exit_worker) {
                    break;
                }
                //puts("4");
                int idx = 1 - item_buf_index;
                // push everything in this block of data
                for (size_t i = 0; i < ItemBufferSize; ++i) {
                    std::tuple<i_type, i_type, d_type>& p = item_buf[idx][i];
                    // sum += std::get<0>(p) + std::get<1>(p) + std::get<2>(p);
                    storage.add_feature_val(std::get<0>(p), std::get<1>(p), std::get<2>(p));
                }
                //puts("5");
                {
                    boost::lock_guard<boost::mutex> lock(work_done_mux);
                    work_done = true;
                }
                work_done_cond.notify_one();
            }
            // std::cout << sum << std::endl;
        }

        /*!breif buffer inputs*/ 
        void push_feature_val (i_type example_index, i_type feature_index, d_type feature_value) {
            // push new data into buffer
            item_buf[item_buf_index][item_index] = std::make_tuple(example_index, feature_index, feature_value);
            item_index++;
            // switch the buffer
            if (item_index == ItemBufferSize) {
                item_index = 0;
                //puts("1");
                // wait until worker thread finishes
                {
                    boost::unique_lock<boost::mutex> lock(work_done_mux);
                    while (!work_done) {
                        work_done_cond.wait(lock);
                    }
                    work_done = false;
                }
                //puts("2");
                item_buf_index = 1 - item_buf_index; 
                // signal the worker thread that new buffer is available
                {
                    boost::lock_guard<boost::mutex> lock(work_ready_mux);
                    work_ready = true;
                }
                work_ready_cond.notify_one();
            }
        }

        /*!brief process one character of the input. Maintains state. */
        inline bool process_one(char* pos) {
            char c = *pos;
            switch(state) {
                case STATE_FEAT:
                    if (c == ':') {
                        state = STATE_VAL;
                        cur_feature_id = positive_atoi(parse_buf.c_str());
                        parse_buf.clear();
                    }
                    else {
                        parse_buf.push_back(c);
                    }
                    break;
                case STATE_VAL:
                    parse_buf.push_back(c);
                    if (unlikely(c == '\n' || c == '\r')) {
                        state = STATE_NEW;
                        cur_feature_value = strtof(parse_buf.c_str());
                        parse_buf.clear();
                        printf("%3ld:%3f\n", cur_feature_id, cur_feature_value);
                        // storage.add_feature_val(cur_example_id, cur_feature_id, cur_feature_value);
                        push_feature_val(cur_example_id, cur_feature_id, cur_feature_value);
                        /* we have processed a full line */
                        return true;
                    }
                    else if (c <= ' ') { // space or tab ('\n' handled above)
                        state = STATE_BLANK;
                        cur_feature_value = strtof(parse_buf.c_str());
                        printf("%3ld:%3f, ", cur_feature_id, cur_feature_value);
                        // storage.add_feature_val(cur_example_id, cur_feature_id, cur_feature_value);
                        push_feature_val(cur_example_id, cur_feature_id, cur_feature_value);
                    }
                    break;
                case STATE_BLANK:
                    if (c > ' ') { // not space, tab and '\n'
                        state = STATE_FEAT;
                        parse_buf.clear();
                        parse_buf.push_back(c);
                    }
                    else if (unlikely(c == '\n' || c == '\r')) {
                        state = STATE_NEW;
                        printf("\n");
                        parse_buf.clear();
                        /* we have processed a full line */
                        return true;
                    }
                    break;
                case STATE_NEW:
                    if (isblank(c) && parse_buf.size()) {
                        state = STATE_BLANK;
                        cur_example_id++;
                        cur_y_value = strtof(parse_buf.c_str());
                        printf("Example %3ld, y=%3f, ", cur_example_id, cur_y_value);
                        storage.add_y(cur_y_value);
                    }
                    else if (c > ' ') { // not space, tab and '\n'
                        parse_buf.push_back(c);
                    }
                    break;
            }
            return false;
        }

        /* !brief process a buffer of whole lines */
        void process_lines(char* lstart, char* lend) {
            char* convert_start;
            while (lstart < lend) {
                /* skip any white space, empty lines, etc */
                if (*lstart <= ' ') {
                    lstart++;
                    continue;
                }
                /* read the label */
                convert_start = lstart;
                /* jump to next white space */
                while(*lstart > ' ') lstart++;
                /* convert the label */
                double label = strtof(convert_start);
                cur_example_id++;
                storage.add_y(label);
                printf("Example %3ld, y=%3f, ", cur_example_id, label);
                /* skip white space */
                while(isblank(*lstart)) lstart++;
                /* process feature index:value pairs */
                while(*lstart != '\n' && *lstart != '\r') {
                    /* read feature ID */
                    convert_start = lstart;
                    /* jump to the next char after ':' */
                    while(*lstart++ != ':');
                    /* convert feature index */
                    size_t index = positive_atoi(convert_start);
                    /* read the feature value */
                    convert_start = lstart;
                    /* jump to next white space */
                    while(*lstart > ' ') lstart++;
                    /* convert feature value */
                    double val = strtof(convert_start);
                    /* skip white space */
                    while(isblank(*lstart)) lstart++;
                    // storage.add_feature_val(cur_example_id, index, val);
                    push_feature_val(cur_example_id, index, val);
                    printf("%3ld:%3f, ", index, val);
                }
                printf("\n");
            }
        }

        /*!brief process a chunk of data */
        void process(char* start_ptr, size_t size) {
            // for(int i = 0; i < size; ++i) putchar(start_ptr[i]);
            /* stop_ptr points after the last valid char */
            char* stop_ptr = start_ptr + size;
            /* Find the last complete line */
            /* lend points to the last '\n', if found */
            /* or start_ptr, if no '\n' is found */
            char* lend_ptr = stop_ptr;
            while (lend_ptr > start_ptr) {
                char c = *(--lend_ptr);
                if (c == '\r' || c == '\n')
                    break;
            }
            /* process the remaining of the last line */
            /* p will be pointed to the beginning of a new line*/
            printf("\nprocess begin\n");
            char* p = start_ptr;
            if (state != STATE_NEW) {
                while(p < stop_ptr)
                    if (process_one(p++))
                        break;
            }
            /* process whole lines */
            printf("\nprocess lines\n");
            process_lines(p, lend_ptr);
            /* process the remaining characters, and maintain state */
            printf("\nprocess remining\n");
            if (p != stop_ptr) { // when no new line in this chunk, p == stop_ptr
                while (lend_ptr < stop_ptr) {
                    process_one(lend_ptr++);
                }
            }
        }

        /*!brief libuv read callback */
        static void on_read(uv_fs_t *req) {
            svm_reader* reader = static_cast<svm_reader*>(req->data);
            // printf("on_read() called\n");
            if (req->result < 0) {
                fprintf(stderr, "Read error: %s\n", uv_strerror(req->result));
            }
            else if (req->result == 0) {
                uv_fs_t close_req;
                printf("closing file\n");
                uv_fs_close(uv_default_loop(), &close_req, reader->fd, NULL);
            }
            else if (req->result > 0) {
                char* buf = reader->iovs[reader->iov_index].base;
                size_t bytes = req->result;
                // request next round;
                printf("\n\n%ld bytes read. requesting more file\n", req->result);
                // switch buffer and request the next block
                reader->iov_index = reader->iov_index ^ 0x1;
                uv_fs_read(uv_default_loop(), &reader->read_req, reader->fd, &reader->iovs[reader->iov_index], 1, -1, svm_reader::on_read);
                printf("processing...\n\n");
                reader->process(buf, bytes);
                printf("\n\ndone.\n\n");
            }
        }
};

