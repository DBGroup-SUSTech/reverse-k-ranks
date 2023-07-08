//
// Created by bianzheng on 2023/5/16.
//

#ifndef TA_TIMERECORD_HPP
#define TA_TIMERECORD_HPP
namespace mips {
    class TimeRecord {
        std::chrono::steady_clock::time_point time_begin;
    public:
        TimeRecord() {
            time_begin = std::chrono::steady_clock::now();
        }

        float get_elapsed_time_second() {
            std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
            std::chrono::duration<float> diff = time_end - time_begin;
            return diff.count();
        }

        void reset() {
            time_begin = std::chrono::steady_clock::now();
        }

    };
}
#endif //TA_TIMERECORD_HPP
