/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 05:37:39 PM CET
 */

#ifndef TIMER_HPP
#define TIMER_HPP

#include <deque>
#include <chrono>

class timer {
  public:
    timer() = default;
    static timer& get() {
        static timer t;
        return t;
    }
    void start() { time_points.push_back(std::chrono::steady_clock::now()); }
    int stop() {
        std::chrono::steady_clock::time_point last = time_points.back();
        time_points.pop_back();
        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - last)
            .count();
    }

  private:
    std::deque<std::chrono::steady_clock::time_point> time_points;
};

#define TIC() timer::get().start()
#define TOC() timer::get().stop()
#define TAC() std::cout << timer::get() << std::endl

#endif
