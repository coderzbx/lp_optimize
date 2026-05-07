#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Config {
    std::string input_path;
    std::string output_path;
    double fs = 2000.0;
    double decimate_to = 100.0;
    bool no_decimate = false;
    bool use_anc = true;
};

struct InputData {
    std::vector<double> timestamps_s;
    std::vector<double> elevation_m;
    std::vector<double> accel_z;
};

struct IdleResult {
    std::vector<double> elevation_m;
    std::vector<double> fluctuation_m;
    double fs = 0.0;
};

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

bool is_number(const std::string& token) {
    const std::string s = trim(token);
    if (s.empty()) {
        return false;
    }
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != nullptr && *end == '\0';
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(trim(field));
    }
    if (!line.empty() && line.back() == ',') {
        fields.emplace_back();
    }
    return fields;
}

double parse_double(const std::string& token, const std::string& ctx) {
    try {
        size_t pos = 0;
        const double value = std::stod(trim(token), &pos);
        if (pos != trim(token).size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("failed to parse number in " + ctx + ": " + token);
    }
}

double compute_median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    const double hi = values[mid];
    if (values.size() % 2 == 1) {
        return hi;
    }
    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
        values.begin() + static_cast<std::ptrdiff_t>(mid)
    );
    return 0.5 * (values[mid - 1] + hi);
}

double nanmean(const std::vector<double>& x) {
    double sum = 0.0;
    size_t count = 0;
    for (double v : x) {
        if (!std::isnan(v)) {
            sum += v;
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

double compute_std(const std::vector<double>& x) {
    if (x.empty()) {
        return 0.0;
    }
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    double acc = 0.0;
    for (double v : x) {
        const double d = v - mean;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(x.size()));
}

size_t mirrored_index(long long idx, size_t n) {
    if (n == 0) {
        return 0;
    }
    while (idx < 0 || idx >= static_cast<long long>(n)) {
        if (idx < 0) {
            idx = -idx;
        } else {
            idx = 2LL * static_cast<long long>(n) - idx - 2LL;
        }
    }
    return static_cast<size_t>(idx);
}

std::vector<double> reflect_pad(const std::vector<double>& x, size_t pad) {
    if (x.empty() || pad == 0) {
        return x;
    }
    std::vector<double> out;
    out.reserve(x.size() + 2 * pad);
    for (size_t i = 0; i < pad; ++i) {
        out.push_back(x[mirrored_index(-static_cast<long long>(pad) + static_cast<long long>(i), x.size())]);
    }
    out.insert(out.end(), x.begin(), x.end());
    for (size_t i = 0; i < pad; ++i) {
        out.push_back(x[mirrored_index(static_cast<long long>(x.size()) + static_cast<long long>(i), x.size())]);
    }
    return out;
}

std::vector<double> unpad(const std::vector<double>& x, size_t pad) {
    if (pad == 0 || x.size() <= 2 * pad) {
        return x;
    }
    return std::vector<double>(x.begin() + static_cast<std::ptrdiff_t>(pad), x.end() - static_cast<std::ptrdiff_t>(pad));
}

struct Biquad {
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;

    std::vector<double> filter(const std::vector<double>& x) const {
        std::vector<double> y(x.size(), 0.0);
        double x1 = 0.0;
        double x2 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            const double xi = x[i];
            const double yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            y[i] = yi;
            x2 = x1;
            x1 = xi;
            y2 = y1;
            y1 = yi;
        }
        return y;
    }
};

Biquad make_lowpass(double fs, double cutoff_hz, double q) {
    const double omega = 2.0 * kPi * cutoff_hz / fs;
    const double sinw = std::sin(omega);
    const double cosw = std::cos(omega);
    const double alpha = sinw / (2.0 * q);
    const double a0 = 1.0 + alpha;
    Biquad bq;
    bq.b0 = ((1.0 - cosw) * 0.5) / a0;
    bq.b1 = (1.0 - cosw) / a0;
    bq.b2 = ((1.0 - cosw) * 0.5) / a0;
    bq.a1 = (-2.0 * cosw) / a0;
    bq.a2 = (1.0 - alpha) / a0;
    return bq;
}

Biquad make_highpass(double fs, double cutoff_hz, double q) {
    const double omega = 2.0 * kPi * cutoff_hz / fs;
    const double sinw = std::sin(omega);
    const double cosw = std::cos(omega);
    const double alpha = sinw / (2.0 * q);
    const double a0 = 1.0 + alpha;
    Biquad bq;
    bq.b0 = ((1.0 + cosw) * 0.5) / a0;
    bq.b1 = (-(1.0 + cosw)) / a0;
    bq.b2 = ((1.0 + cosw) * 0.5) / a0;
    bq.a1 = (-2.0 * cosw) / a0;
    bq.a2 = (1.0 - alpha) / a0;
    return bq;
}

std::vector<double> zero_phase_filter(const std::vector<double>& x, const std::vector<Biquad>& sections) {
    if (x.empty()) {
        return {};
    }
    const size_t pad = std::min<size_t>(std::max<size_t>(6, sections.size() * 6), x.size() > 1 ? x.size() - 1 : 0);
    std::vector<double> y = reflect_pad(x, pad);
    for (const auto& sec : sections) {
        y = sec.filter(y);
    }
    std::reverse(y.begin(), y.end());
    for (const auto& sec : sections) {
        y = sec.filter(y);
    }
    std::reverse(y.begin(), y.end());
    return unpad(y, pad);
}

std::vector<double> highpass_detrend(const std::vector<double>& x, double fs, double fc) {
    if (fc <= 0.0 || x.empty()) {
        return x;
    }
    const double nyq = 0.5 * fs;
    const double cutoff = std::min(fc, 0.99 * nyq);
    return zero_phase_filter(x, {make_highpass(fs, cutoff, std::sqrt(0.5))});
}

std::vector<double> lowpass_filter(const std::vector<double>& x, double fs, double fc) {
    if (x.empty()) {
        return {};
    }
    const double nyq = 0.5 * fs;
    const double cutoff = std::min(fc, 0.99 * nyq);
    return zero_phase_filter(
        x,
        {
            make_lowpass(fs, cutoff, 0.541196100146197),
            make_lowpass(fs, cutoff, 1.306562964876377),
        }
    );
}

std::vector<double> interpolate_gaps(const std::vector<double>& x) {
    std::vector<double> out = x;
    std::vector<size_t> good_idx;
    good_idx.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isnan(x[i])) {
            good_idx.push_back(i);
        }
    }
    if (good_idx.size() < 2) {
        return out;
    }
    size_t prev = good_idx.front();
    for (size_t i = 0; i < prev; ++i) {
        out[i] = x[prev];
    }
    for (size_t g = 1; g < good_idx.size(); ++g) {
        const size_t left = good_idx[g - 1];
        const size_t right = good_idx[g];
        if (right - left > 1) {
            for (size_t i = left + 1; i < right; ++i) {
                const double alpha = static_cast<double>(i - left) / static_cast<double>(right - left);
                out[i] = x[left] * (1.0 - alpha) + x[right] * alpha;
            }
        }
    }
    for (size_t i = good_idx.back() + 1; i < x.size(); ++i) {
        out[i] = x[good_idx.back()];
    }
    return out;
}

std::vector<double> hampel_filter(const std::vector<double>& x, int window, double n_sigma) {
    if (x.empty()) {
        return {};
    }
    if (window < 1) {
        return x;
    }
    if (window % 2 == 0) {
        ++window;
    }
    const int half = window / 2;
    constexpr double kMadScale = 1.4826;
    std::vector<double> out = x;
    std::vector<double> buf;
    buf.reserve(static_cast<size_t>(window));
    for (size_t i = 0; i < x.size(); ++i) {
        buf.clear();
        for (int j = -half; j <= half; ++j) {
            buf.push_back(x[mirrored_index(static_cast<long long>(i) + j, x.size())]);
        }
        const double med = compute_median(buf);
        for (double& v : buf) {
            v = std::abs(v - med);
        }
        const double mad = compute_median(buf);
        const double threshold = n_sigma * kMadScale * mad;
        if (std::abs(x[i] - med) > threshold) {
            out[i] = med;
        }
    }
    return out;
}

std::vector<double> median_filter(const std::vector<double>& x, int window) {
    if (x.empty() || window <= 1) {
        return x;
    }
    if (window % 2 == 0) {
        ++window;
    }
    const int half = window / 2;
    std::vector<double> out(x.size(), 0.0);
    std::vector<double> buf;
    buf.reserve(static_cast<size_t>(window));
    for (size_t i = 0; i < x.size(); ++i) {
        buf.clear();
        for (int j = -half; j <= half; ++j) {
            buf.push_back(x[mirrored_index(static_cast<long long>(i) + j, x.size())]);
        }
        out[i] = compute_median(buf);
    }
    return out;
}

std::vector<std::vector<double>> identity_matrix(size_t n) {
    std::vector<std::vector<double>> m(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        m[i][i] = 1.0;
    }
    return m;
}

std::vector<std::vector<double>> invert_matrix(std::vector<std::vector<double>> a) {
    const size_t n = a.size();
    auto inv = identity_matrix(n);
    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        double best = std::abs(a[col][col]);
        for (size_t row = col + 1; row < n; ++row) {
            const double candidate = std::abs(a[row][col]);
            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }
        if (best < 1e-12) {
            throw std::runtime_error("matrix inversion failed");
        }
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(inv[pivot], inv[col]);
        }
        const double diag = a[col][col];
        for (size_t j = 0; j < n; ++j) {
            a[col][j] /= diag;
            inv[col][j] /= diag;
        }
        for (size_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = a[row][col];
            if (std::abs(factor) < 1e-18) {
                continue;
            }
            for (size_t j = 0; j < n; ++j) {
                a[row][j] -= factor * a[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }
    return inv;
}

std::vector<double> savgol_kernel(int window, int order) {
    if (window % 2 == 0) {
        ++window;
    }
    window = std::max(window, order + 2 + ((order % 2 == 0) ? 1 : 0));
    if (window % 2 == 0) {
        ++window;
    }
    const int half = window / 2;
    const int p = order + 1;
    std::vector<std::vector<double>> ata(static_cast<size_t>(p), std::vector<double>(static_cast<size_t>(p), 0.0));
    for (int row = -half; row <= half; ++row) {
        std::vector<double> powers(static_cast<size_t>(p), 1.0);
        for (int j = 1; j < p; ++j) {
            powers[static_cast<size_t>(j)] = powers[static_cast<size_t>(j - 1)] * static_cast<double>(row);
        }
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
                ata[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
                    powers[static_cast<size_t>(i)] * powers[static_cast<size_t>(j)];
            }
        }
    }
    const auto ata_inv = invert_matrix(ata);
    std::vector<double> kernel(static_cast<size_t>(window), 0.0);
    for (int row = -half; row <= half; ++row) {
        std::vector<double> powers(static_cast<size_t>(p), 1.0);
        for (int j = 1; j < p; ++j) {
            powers[static_cast<size_t>(j)] = powers[static_cast<size_t>(j - 1)] * static_cast<double>(row);
        }
        double coeff = 0.0;
        for (int j = 0; j < p; ++j) {
            coeff += ata_inv[0][static_cast<size_t>(j)] * powers[static_cast<size_t>(j)];
        }
        kernel[static_cast<size_t>(row + half)] = coeff;
    }
    return kernel;
}

std::vector<double> savgol_filter(const std::vector<double>& x, int window, int order) {
    if (x.empty()) {
        return {};
    }
    const auto kernel = savgol_kernel(window, order);
    const int half = static_cast<int>(kernel.size() / 2);
    std::vector<double> out(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        double acc = 0.0;
        for (int j = -half; j <= half; ++j) {
            acc += kernel[static_cast<size_t>(j + half)] *
                   x[mirrored_index(static_cast<long long>(i) + j, x.size())];
        }
        out[i] = acc;
    }
    return out;
}

size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) {
        p <<= 1U;
    }
    return p;
}

void fft(std::vector<std::complex<double>>& a, bool invert) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1U;
        for (; j & bit; bit >>= 1U) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1U) {
        const double ang = 2.0 * kPi / static_cast<double>(len) * (invert ? -1.0 : 1.0);
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                const auto u = a[i + j];
                const auto v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (auto& v : a) {
            v /= static_cast<double>(n);
        }
    }
}

std::vector<double> accel_to_displacement(const std::vector<double>& accel, double fs, double band_lo, double band_hi) {
    if (accel.empty()) {
        return {};
    }
    std::vector<double> demeaned = accel;
    const double mean = nanmean(demeaned);
    for (double& v : demeaned) {
        v -= mean;
    }
    const size_t n = accel.size();
    const size_t nfft = next_power_of_two(n);
    std::vector<std::complex<double>> spec(nfft, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < n; ++i) {
        spec[i] = std::complex<double>(demeaned[i], 0.0);
    }
    fft(spec, false);
    const double nyq_guard = 0.5 * fs * 0.999;
    for (size_t k = 0; k < nfft; ++k) {
        const double f = (k <= nfft / 2) ? (static_cast<double>(k) * fs / static_cast<double>(nfft))
                                         : (static_cast<double>(nfft - k) * fs / static_cast<double>(nfft));
        if (f < band_lo || f > std::min(band_hi, nyq_guard) || f == 0.0) {
            spec[k] = 0.0;
            continue;
        }
        const double omega2 = std::pow(2.0 * kPi * f, 2.0);
        spec[k] = -spec[k] / omega2;
    }
    fft(spec, true);
    std::vector<double> displacement(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        displacement[i] = spec[i].real();
    }
    return displacement;
}

std::pair<std::vector<double>, std::vector<double>> lms_anc(
    const std::vector<double>& primary,
    const std::vector<double>& reference,
    int n_taps,
    double mu,
    double eps = 1e-6
) {
    if (primary.size() != reference.size()) {
        throw std::runtime_error("primary and reference must have the same shape");
    }
    std::vector<double> weights(static_cast<size_t>(n_taps), 0.0);
    std::vector<double> xbuf(static_cast<size_t>(n_taps), 0.0);
    std::vector<double> error(primary.size(), 0.0);
    for (size_t k = 0; k < primary.size(); ++k) {
        for (int i = n_taps - 1; i > 0; --i) {
            xbuf[static_cast<size_t>(i)] = xbuf[static_cast<size_t>(i - 1)];
        }
        xbuf[0] = reference[k];
        double y = 0.0;
        double norm = eps;
        for (int i = 0; i < n_taps; ++i) {
            y += weights[static_cast<size_t>(i)] * xbuf[static_cast<size_t>(i)];
            norm += xbuf[static_cast<size_t>(i)] * xbuf[static_cast<size_t>(i)];
        }
        const double e = primary[k] - y;
        const double gain = mu * e / norm;
        for (int i = 0; i < n_taps; ++i) {
            weights[static_cast<size_t>(i)] += gain * xbuf[static_cast<size_t>(i)];
        }
        error[k] = e;
    }
    return {error, weights};
}

std::vector<double> linspace_indices(size_t n) {
    std::vector<double> out(n, 0.0);
    if (n <= 1) {
        return out;
    }
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return out;
}

std::vector<double> resample_series_axis(const std::vector<double>& values, size_t n_out) {
    if (n_out == 0) {
        return {};
    }
    if (values.empty()) {
        return std::vector<double>(n_out, 0.0);
    }
    if (values.size() == n_out) {
        return values;
    }
    if (values.size() == 1) {
        return std::vector<double>(n_out, values[0]);
    }
    const auto src = linspace_indices(values.size());
    const auto dst = linspace_indices(n_out);
    std::vector<double> out(n_out, 0.0);
    size_t j = 0;
    for (size_t i = 0; i < n_out; ++i) {
        const double x = dst[i];
        while (j + 1 < src.size() && src[j + 1] < x) {
            ++j;
        }
        const size_t j1 = std::min(j + 1, src.size() - 1);
        const double x0 = src[j];
        const double x1 = src[j1];
        if (x1 <= x0) {
            out[i] = values[j];
        } else {
            const double alpha = (x - x0) / (x1 - x0);
            out[i] = values[j] * (1.0 - alpha) + values[j1] * alpha;
        }
    }
    return out;
}

std::pair<std::vector<double>, double> antialias_decimate(const std::vector<double>& x, double fs, double fs_out) {
    if (x.empty() || fs_out >= fs) {
        return {x, fs};
    }
    auto filtered = lowpass_filter(x, fs, std::min(0.45 * fs_out, 40.0));
    const size_t n_out = std::max<size_t>(1, static_cast<size_t>(std::llround(filtered.size() * fs_out / fs)));
    return {resample_series_axis(filtered, n_out), fs_out};
}

std::vector<double> add_vectors(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("shape mismatch");
    }
    std::vector<double> out(a.size(), 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

std::vector<double> idle_residual_cleanup(
    const std::vector<double>& x,
    double fs,
    int residual_hampel_window,
    double residual_hampel_sigma,
    int median_window,
    double smooth_cutoff,
    int savgol_window,
    int savgol_order
) {
    auto y = hampel_filter(x, residual_hampel_window, residual_hampel_sigma);
    y = median_filter(y, median_window);
    y = lowpass_filter(y, fs, smooth_cutoff);
    y = savgol_filter(y, savgol_window, savgol_order);
    return y;
}

IdleResult pipeline_idle_on(
    const std::vector<double>& elevation_m,
    const std::vector<double>& accel_z,
    double fs,
    bool use_anc,
    double decimate_to
) {
    if (elevation_m.size() != accel_z.size()) {
        throw std::runtime_error("elevation and accel_z must have the same shape");
    }
    auto x = interpolate_gaps(elevation_m);
    x = hampel_filter(x, 21, 3.0);
    const auto z_body = accel_to_displacement(accel_z, fs, 0.5, 80.0);
    const double baseline = compute_median(add_vectors(x, z_body));
    x = highpass_detrend(x, fs, 0.05);
    auto x_comp = add_vectors(x, z_body);
    if (use_anc) {
        x_comp = lms_anc(x_comp, z_body, 128, 0.05).first;
    }
    x_comp = idle_residual_cleanup(x_comp, fs, 61, 2.5, 9, 40.0, 41, 3);
    double fs_out = fs;
    if (decimate_to > 0.0 && decimate_to < fs) {
        auto decimated = antialias_decimate(x_comp, fs, decimate_to);
        x_comp = std::move(decimated.first);
        fs_out = decimated.second;
    }
    std::vector<double> elevation(x_comp.size(), 0.0);
    for (size_t i = 0; i < x_comp.size(); ++i) {
        elevation[i] = baseline + x_comp[i];
    }
    return {elevation, x_comp, fs_out};
}

InputData read_input_csv_with_timestamps(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    std::string line;
    bool first_data_row = true;
    size_t row_index = 0;
    InputData data;
    while (std::getline(in, line)) {
        auto fields = split_csv_line(line);
        bool empty = true;
        for (const auto& field : fields) {
            if (!field.empty()) {
                empty = false;
                break;
            }
        }
        if (empty) {
            continue;
        }
        if (first_data_row) {
            first_data_row = false;
            if (fields.size() >= 2 && !is_number(fields[1])) {
                continue;
            }
        }
        if (fields.size() < 3) {
            throw std::runtime_error(path + ": expected at least 3 columns");
        }
        data.timestamps_s.push_back(
            is_number(fields[0]) ? parse_double(fields[0], path) : static_cast<double>(row_index)
        );
        data.elevation_m.push_back(parse_double(fields[1], path) * 1e-3);
        data.accel_z.push_back(parse_double(fields[2], path));
        ++row_index;
    }
    if (data.elevation_m.empty()) {
        throw std::runtime_error("no data rows found in " + path);
    }
    return data;
}

void write_result_csv(
    const std::string& path,
    const std::vector<double>& timestamps_s,
    const std::vector<double>& elevation_m,
    const std::vector<double>& fluctuation_m,
    double fs
) {
    if (timestamps_s.size() != elevation_m.size() || elevation_m.size() != fluctuation_m.size()) {
        throw std::runtime_error("output vectors must have the same shape");
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out << "timestamp_s,elevation_mm,fluctuation_mm,t_s\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < elevation_m.size(); ++i) {
        const double t = static_cast<double>(i) / fs;
        out << timestamps_s[i] << ',' << elevation_m[i] * 1e3 << ',' << fluctuation_m[i] * 1e3 << ',' << t
            << '\n';
    }
}

Config parse_args(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "usage: problem2_idle_on <C040.csv> <Result_C040.csv> [--fs 2000] [--decimate-to 100] [--no-decimate] [--no-anc]"
        );
    }
    Config cfg;
    cfg.input_path = argv[1];
    cfg.output_path = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fs") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--fs requires a value");
            }
            cfg.fs = parse_double(argv[++i], "--fs");
        } else if (arg == "--decimate-to") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--decimate-to requires a value");
            }
            cfg.decimate_to = parse_double(argv[++i], "--decimate-to");
        } else if (arg == "--no-decimate") {
            cfg.no_decimate = true;
        } else if (arg == "--no-anc") {
            cfg.use_anc = false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);
        const auto input = read_input_csv_with_timestamps(cfg.input_path);
        const double decimate_to = cfg.no_decimate ? cfg.fs : cfg.decimate_to;
        const auto result = pipeline_idle_on(input.elevation_m, input.accel_z, cfg.fs, cfg.use_anc, decimate_to);
        const auto timestamps_out = resample_series_axis(input.timestamps_s, result.elevation_m.size());
        write_result_csv(
            cfg.output_path,
            timestamps_out,
            result.elevation_m,
            result.fluctuation_m,
            result.fs
        );
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "[idle-on-cpp] n=" << result.elevation_m.size()
                  << "  std=" << (compute_std(result.fluctuation_m) * 1e3) << " mm"
                  << "  wrote " << cfg.output_path
                  << "  (fs_out=" << result.fs << " Hz)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
