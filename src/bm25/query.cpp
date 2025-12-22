// BM25 Query with Multi-threading
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

const double K1 = 1.2, B = 0.75;
const int BLOCK_SIZE = 128;

const unordered_set<string> STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "now"
};

class PorterStemmer {
    bool isC(const string& w, int i) {
        char c = w[i];
        return c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' &&
               (c != 'y' || i == 0 || !isC(w, i - 1));
    }

    int m(const string& w) {
        int m = 0, i = 0, n = w.size();
        while (i < n && isC(w, i)) i++;
        while (i < n) {
            while (i < n && !isC(w, i)) i++;
            if (i >= n) break;
            m++;
            while (i < n && isC(w, i)) i++;
        }
        return m;
    }

    bool hasV(const string& w) {
        for (int i = 0; i < (int)w.size(); i++)
            if (!isC(w, i)) return true;
        return false;
    }

    bool dblC(const string& w) {
        int n = w.size();
        return n >= 2 && w[n-1] == w[n-2] && isC(w, n-1);
    }

    bool cvc(const string& w) {
        int n = w.size();
        return n >= 3 && isC(w, n-1) && !isC(w, n-2) && isC(w, n-3) &&
               w[n-1] != 'w' && w[n-1] != 'x' && w[n-1] != 'y';
    }

    bool ends(const string& w, const string& s) {
        return s.size() <= w.size() &&
               w.compare(w.size() - s.size(), s.size(), s) == 0;
    }

    string rep(const string& w, const string& s, const string& r) {
        return w.substr(0, w.size() - s.size()) + r;
    }

public:
    string stem(const string& w) {
        if (w.size() <= 2) return w;
        string s = w;

        // Step 1a
        if (ends(s, "sses")) s = rep(s, "sses", "ss");
        else if (ends(s, "ies")) s = rep(s, "ies", "i");
        else if (!ends(s, "ss") && ends(s, "s")) s = s.substr(0, s.size() - 1);

        // Step 1b
        bool f = false;
        if (ends(s, "eed")) {
            if (m(s.substr(0, s.size() - 3)) > 0) s = rep(s, "eed", "ee");
        } else if (ends(s, "ed")) {
            string t = s.substr(0, s.size() - 2);
            if (hasV(t)) { s = t; f = true; }
        } else if (ends(s, "ing")) {
            string t = s.substr(0, s.size() - 3);
            if (hasV(t)) { s = t; f = true; }
        }

        if (f) {
            if (ends(s, "at") || ends(s, "bl") || ends(s, "iz")) s += "e";
            else if (dblC(s) && s.back() != 'l' && s.back() != 's' && s.back() != 'z')
                s = s.substr(0, s.size() - 1);
            else if (m(s) == 1 && cvc(s)) s += "e";
        }

        // Step 1c
        if (ends(s, "y") && hasV(s.substr(0, s.size() - 1)))
            s = rep(s, "y", "i");

        // Step 2
        const char* s2[][2] = {
            {"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"},
            {"anci", "ance"}, {"izer", "ize"}, {"abli", "able"},
            {"alli", "al"}, {"entli", "ent"}, {"eli", "e"}, {"ousli", "ous"},
            {"ization", "ize"}, {"ation", "ate"}, {"ator", "ate"},
            {"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
            {"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"}, {"biliti", "ble"}
        };
        for (auto& p : s2) {
            if (ends(s, p[0])) {
                string t = s.substr(0, s.size() - strlen(p[0]));
                if (m(t) > 0) s = t + p[1];
                break;
            }
        }

        // Step 3
        const char* s3[][2] = {
            {"icate", "ic"}, {"ative", ""}, {"alize", "al"},
            {"iciti", "ic"}, {"ical", "ic"}, {"ful", ""}, {"ness", ""}
        };
        for (auto& p : s3) {
            if (ends(s, p[0])) {
                string t = s.substr(0, s.size() - strlen(p[0]));
                if (m(t) > 0) s = t + p[1];
                break;
            }
        }

        // Step 4
        const char* s4[] = {
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize"
        };
        for (auto& x : s4) {
            if (ends(s, x)) {
                string t = s.substr(0, s.size() - strlen(x));
                if (m(t) > 1) {
                    if (strcmp(x, "ion") == 0) {
                        if (!t.empty() && (t.back() == 's' || t.back() == 't'))
                            s = t;
                    } else {
                        s = t;
                    }
                }
                break;
            }
        }

        // Step 5
        if (ends(s, "e")) {
            string t = s.substr(0, s.size() - 1);
            int mm = m(t);
            if (mm > 1 || (mm == 1 && !cvc(t))) s = t;
        }
        if (m(s) > 1 && dblC(s) && s.back() == 'l')
            s = s.substr(0, s.size() - 1);

        return s;
    }
} stemmer;

// Globals
unordered_map<string, tuple<long long, int, int, int>> lexicon;
vector<int> lastDocIDs, docIDSizes, freqSizes;
unordered_map<int, int> docLengths;
unordered_map<int, string> docIdMap;
int totalDocs = 0;
double avgLen = 0;
mutex outMutex;

int vb_decode(const unsigned char* d, int& o) {
    int n = 0, s = 0;
    unsigned char b;
    do {
        b = d[o++];
        n |= (b & 0x7F) << s;
        s += 7;
    } while (b & 0x80);
    return n;
}

vector<string> tokenize(const string& s) {
    vector<string> t;
    string w;
    for (char c : s) {
        if (isalnum(c)) {
            w += tolower(c);
        } else if (!w.empty()) {
            if (w.size() > 1 && !STOPWORDS.count(w))
                t.push_back(stemmer.stem(w));
            w.clear();
        }
    }
    if (!w.empty() && w.size() > 1 && !STOPWORDS.count(w))
        t.push_back(stemmer.stem(w));
    return t;
}

class InvList {
    ifstream f;
    long long off;
    int sb, np, bi, pi;
    vector<int> docs, freqs;
    bool done = false;

    void load() {
        docs.clear();
        freqs.clear();
        if (bi >= sb + (np + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            done = true;
            return;
        }

        long long o = off;
        for (int i = sb; i < bi; i++)
            o += 4 + docIDSizes[i] + 4 + freqSizes[i];

        f.seekg(o);
        int ds;
        f.read((char*)&ds, 4);
        vector<unsigned char> d(ds);
        f.read((char*)d.data(), ds);

        int fs;
        f.read((char*)&fs, 4);
        vector<unsigned char> fr(fs);
        f.read((char*)fr.data(), fs);

        int p = 0;
        while (p < ds) docs.push_back(vb_decode(d.data(), p));
        for (size_t i = 1; i < docs.size(); i++) docs[i] += docs[i-1];

        p = 0;
        while (p < fs) freqs.push_back(vb_decode(fr.data(), p));
        pi = 0;
    }

public:
    InvList(const string& t) {
        auto it = lexicon.find(t);
        if (it == lexicon.end()) {
            done = true;
            return;
        }
        f.open("index/inverted_index.bin", ios::binary);
        off = get<0>(it->second);
        sb = get<1>(it->second);
        np = get<2>(it->second);
        bi = sb;
        load();
    }

    bool nextGEQ(int tgt) {
        if (done) return false;
        while (bi < sb + (np + BLOCK_SIZE - 1) / BLOCK_SIZE) {
            if (lastDocIDs[bi] >= tgt) {
                if (docs.empty() || pi >= (int)docs.size()) load();
                break;
            }
            bi++;
        }
        if (done) return false;
        while (pi < (int)docs.size()) {
            if (docs[pi] >= tgt) return true;
            pi++;
        }
        bi++;
        load();
        return nextGEQ(tgt);
    }

    bool has() { return !done && pi < (int)docs.size(); }
    int doc() { return docs[pi]; }
    int freq() { return freqs[pi]; }
    void next() { pi++; }
};

double bm25(int tf, int dl, int df) {
    return log((totalDocs - df + 0.5) / (df + 0.5)) *
           ((tf * (K1 + 1)) / (tf + K1 * (1 - B + B * (dl / avgLen))));
}

vector<pair<int, double>> query(const vector<string>& terms) {
    static thread_local vector<double> sc(totalDocs, 0);
    static thread_local vector<int> touched;

    for (auto& t : terms) {
        auto it = lexicon.find(t);
        if (it == lexicon.end()) continue;
        int df = get<3>(it->second);
        InvList l(t);
        l.nextGEQ(0);
        while (l.has()) {
            int d = l.doc(), f = l.freq();
            if (sc[d] == 0) touched.push_back(d);
            sc[d] += bm25(f, docLengths[d], df);
            l.next();
        }
    }

    vector<pair<int, double>> r;
    for (int d : touched) r.emplace_back(d, sc[d]);
    touched.clear();
    for (auto& p : r) sc[p.first] = 0;

    sort(r.begin(), r.end(), [](auto& a, auto& b) { return a.second > b.second; });
    if (r.size() > 1000) r.resize(1000);
    return r;
}

bool loadIndex() {
    ifstream f("index/lexicon.txt");
    if (!f) return false;
    string t;
    long long o;
    int sb, np, df;
    while (f >> t >> o >> sb >> np >> df)
        lexicon[t] = {o, sb, np, df};
    f.close();

    ifstream m("index/metadata.bin", ios::binary);
    int n;
    m.read((char*)&n, 4);
    lastDocIDs.resize(n);
    docIDSizes.resize(n);
    freqSizes.resize(n);
    m.read((char*)lastDocIDs.data(), n * 4);
    m.read((char*)docIDSizes.data(), n * 4);
    m.read((char*)freqSizes.data(), n * 4);
    m.close();

    ifstream d("index/doc_lengths.txt");
    int id, l;
    while (d >> id >> l) {
        docLengths[id] = l;
        avgLen += l;
        totalDocs++;
    }
    avgLen /= totalDocs;

    ifstream p("index/page_table.txt");
    string ext;
    while (p >> id >> ext) docIdMap[id] = ext;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <queries.tsv>\n";
        return 1;
    }

    if (!loadIndex()) {
        cerr << "Index load failed\n";
        return 1;
    }

    ifstream q(argv[1]);
    vector<string> lines;
    string line;
    while (getline(q, line)) lines.push_back(line);
    q.close();

    string outFile = fs::path(argv[1]).stem().string() + "_results.txt";
    ofstream out(outFile);

    atomic<size_t> idx(0);
    vector<thread> threads;

    auto worker = [&]() {
        vector<string> local;
        while (true) {
            size_t i = idx++;
            if (i >= lines.size()) break;

            stringstream ss(lines[i]);
            string id, text;
            getline(ss, id, '\t');
            getline(ss, text);

            auto res = query(tokenize(text));
            int rk = 1;
            for (auto& p : res)
                local.push_back(id + " Q0 " + docIdMap[p.first] + " " +
                               to_string(rk++) + " " + to_string(p.second) + " bm25");
        }

        lock_guard<mutex> lock(outMutex);
        for (auto& s : local) out << s << "\n";
    };

    for (unsigned i = 0; i < thread::hardware_concurrency(); i++)
        threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    cout << "Done. Results: " << outFile << "\n";
    return 0;
}
