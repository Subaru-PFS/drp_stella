#include <chrono>
#include <map>

namespace pfs {
namespace drp {
namespace stella {
namespace utils {


/// A timer suitable for timing small blocks of code.
class Timer {
  public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    /// Ctor
    Timer() {
        reset();
    }

    // Copy not permitted
    Timer(Timer const &) = delete;
    Timer & operator=(Timer const &) = delete;
    // Move permitted
    Timer(Timer &&) = default;
    Timer & operator=(Timer &&) = default;

    ~Timer() = default;

    /// Reset the timer
    void reset() {
        _start = Clock::now();
    }

    /// Return the elapsed time in seconds
    double elapsed() const {
        auto const end = Clock::now();
        return std::chrono::duration<double>(end - _start).count();
    }

  private:
    TimePoint _start;  ///< Start time
};


/// A set of timers, each with a name.
///
/// This class is useful for timing multiple blocks of code, each given a
/// different symbolic name. The total elapsed time is also recorded.
///
/// The timers are started by calling `context(name)` and stopped when the
/// returned object is destroyed. To use:
///
///    TimerSet timers;
///    ... {  // some block of code
///        auto timer = timers.context("one");
///        ...  // code to time
///    }  // "one" timer stops here
///    ... {  // another block of code
///        auto timer = timers.context("two");
///        ...  // code to time
///    }  // "two" timer stops here
///
/// When the `TimerSet` object goes out of scope, the elapsed times are printed
/// to `std::cerr` (if the `printWhenDone` constructor argument is `true`;
/// otherwise you can retrieve the elapsed times with the `elapsed` and/or
/// `getElapsed` methods).
class TimerSet {
public:
    using ElapsedMap = std::map<std::string, double>;

    TimerSet(bool printWhenDone = true)
      : _printWhenDone(printWhenDone) {
        reset();
    }

    // Copy not permitted
    TimerSet(TimerSet const &) = delete;
    TimerSet & operator=(TimerSet const &) = delete;
    // Move permitted
    TimerSet(TimerSet &&) = default;
    TimerSet & operator=(TimerSet &&) = default;

    ~TimerSet() {
        if (!_printWhenDone) {
            return;
        }
        if (_elapsed.size() > 0) {
            std::cerr << "Timers:" << std::endl;
            for (auto const & pair : _elapsed) {
                std::cerr << "  " << pair.first << ": " << pair.second << " sec" << std::endl;
            }
        }
        std::cerr << "Total elapsed time: " << _total.elapsed() << " sec" << std::endl;
    }

    /// Reset the timers
    void reset() {
        _elapsed.clear();
        _total.reset();
    }

    /// RAII timer context
    ///
    /// Updates the elapsed time in the destructor.
    class TimerContext {
      public:
        TimerContext(ElapsedMap::reference elapsed) : _elapsed(elapsed) {}

        // Copy not permitted
        TimerContext(TimerContext const &) = delete;
        TimerContext & operator=(TimerContext const &) = delete;
        // Move permitted
        TimerContext(TimerContext &&) = default;
        TimerContext & operator=(TimerContext &&) = default;

        /// Destructor: update the elapsed time
        ~TimerContext() {
            _elapsed.second += _timer.elapsed();
        }

      private:
        Timer _timer;  ///< Timer counting the seconds
        ElapsedMap::reference _elapsed;  ///< Elapsed time to update on destruction
    };

    TimerContext context(std::string const & name) {
        auto iter = _elapsed.find(name);
        if (iter == _elapsed.end()) {
            iter = _elapsed.insert({name, 0.0}).first;
        }
        return TimerContext(*iter);
    }

    //@{
    /// Return the elapsed time for a given timer
    double elapsed(std::string const & name) const {
        return _elapsed.at(name);
    }
    double elapsed() const {
        return _total.elapsed();
    }
    //@}

    /// Return the elapsed times for all timers
    ElapsedMap const & getElapsed() const {
        return _elapsed;
    }

  private:
    bool _printWhenDone;  ///< Print the elapsed times when the object goes out of scope?
    Timer _total;  ///< Total elapsed time
    ElapsedMap _elapsed;  ///< Elapsed times for each timer
};


}}}}  // namespace pfs::drp::stella::utils
