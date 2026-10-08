#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

enum class BackendMode { Cpu, Mixed };
enum class QuantMode { Float, Int8 };

struct CliConfig {
    std::string model_path;
    std::string prompt;
    std::int64_t max_tokens = 16;
    std::uint64_t seed = 0;
    BackendMode backend = BackendMode::Cpu;
    QuantMode quant = QuantMode::Float;
    bool cache = true;
    bool dry_run = false;
};

std::string json_quote(const std::string& value) {
    std::string result{"\""};
    for (const unsigned char character : value) {
        switch (character) {
        case '\\': result += "\\\\"; break;
        case '\"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (character < 0x20U) throw std::invalid_argument("control bytes are not supported in CLI strings");
            result.push_back(static_cast<char>(character));
        }
    }
    result += '"';
    return result;
}

std::int64_t parse_positive(const std::string& value, const char* name) {
    std::size_t parsed = 0;
    long long number = 0;
    try {
        number = std::stoll(value, &parsed, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    if (parsed != value.size() || number <= 0) throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return number;
}

std::uint64_t parse_seed(const std::string& value) {
    if (value.empty() || value.front() == '-') throw std::invalid_argument("--seed must be an unsigned integer");
    std::size_t parsed = 0;
    unsigned long long number = 0;
    try {
        number = std::stoull(value, &parsed, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument("--seed must be an unsigned integer");
    }
    if (parsed != value.size()) throw std::invalid_argument("--seed must be an unsigned integer");
    return number;
}

const char* name(BackendMode mode) { return mode == BackendMode::Cpu ? "cpu" : "mixed"; }
const char* name(QuantMode mode) { return mode == QuantMode::Float ? "float" : "int8"; }

void print_help(std::ostream& output) {
    output << "usage: llm_cli --model FILE --prompt TEXT [--max-tokens N] [--backend cpu|mixed] "
              "[--quant float|int8] [--cache on|off] [--seed N] [--dry-run]\n"
              "\n"
              "Batch is fixed at one. Generation is deterministic greedy decoding; --seed is recorded "
              "for reproducibility and has no effect until an optional sampling mode exists.\n";
}

CliConfig parse_args(int argc, char** argv) {
    CliConfig config;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&](const char* option) -> std::string {
            if (++index >= argc) throw std::invalid_argument(std::string(option) + " requires a value");
            return argv[index];
        };
        if (argument == "--help" || argument == "-h") {
            print_help(std::cout);
            std::exit(0);
        } else if (argument == "--model") {
            config.model_path = value("--model");
        } else if (argument == "--prompt") {
            config.prompt = value("--prompt");
        } else if (argument == "--max-tokens") {
            config.max_tokens = parse_positive(value("--max-tokens"), "--max-tokens");
        } else if (argument == "--seed") {
            config.seed = parse_seed(value("--seed"));
        } else if (argument == "--backend") {
            const auto& selected = value("--backend");
            if (selected == "cpu") config.backend = BackendMode::Cpu;
            else if (selected == "mixed") config.backend = BackendMode::Mixed;
            else throw std::invalid_argument("--backend must be cpu or mixed");
        } else if (argument == "--quant") {
            const auto& selected = value("--quant");
            if (selected == "float") config.quant = QuantMode::Float;
            else if (selected == "int8") config.quant = QuantMode::Int8;
            else throw std::invalid_argument("--quant must be float or int8");
        } else if (argument == "--cache") {
            const auto& selected = value("--cache");
            if (selected == "on") config.cache = true;
            else if (selected == "off") config.cache = false;
            else throw std::invalid_argument("--cache must be on or off");
        } else if (argument == "--dry-run") {
            config.dry_run = true;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (config.model_path.empty()) throw std::invalid_argument("--model is required");
    return config;
}

void print_config(const CliConfig& config) {
    std::cout << "{\"status\":\"configured\",\"model\":" << json_quote(config.model_path)
              << ",\"prompt\":" << json_quote(config.prompt)
              << ",\"max_tokens\":" << config.max_tokens
              << ",\"backend\":" << json_quote(name(config.backend))
              << ",\"quant\":" << json_quote(name(config.quant))
              << ",\"cache\":" << (config.cache ? "true" : "false")
              << ",\"seed\":" << config.seed
              << ",\"sampling\":\"greedy\"}" << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto config = parse_args(argc, argv);
        print_config(config);
        if (!config.dry_run)
            throw std::runtime_error("generation is unavailable until Stage 17-C2; use --dry-run for configuration validation");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "{\"status\":\"error\",\"message\":" << json_quote(error.what()) << "}" << '\n';
        return 1;
    }
}
