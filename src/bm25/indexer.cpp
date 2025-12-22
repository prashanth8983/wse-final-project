// BM25 Indexer with Porter Stemmer
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <cstring>

using namespace std;

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

void writePartial(const vector<tuple<string, int, int>>& p, int run) {
    ofstream out("partial/run_" + to_string(run) + ".bin", ios::binary);
    for (auto& [term, doc, freq] : p) {
        int len = term.size();
        out.write((char*)&len, 4);
        out.write(term.data(), len);
        out.write((char*)&doc, 4);
        out.write((char*)&freq, 4);
    }
    cerr << "Saved run_" << run << ".bin (" << p.size() << " postings)\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input.tsv>\n";
        return 1;
    }

    system("mkdir -p partial");
    system("mkdir -p index");

    ifstream file(argv[1]);
    if (!file) {
        cerr << "Cannot open " << argv[1] << "\n";
        return 1;
    }

    // Load subset filter
    unordered_set<string> allowed;
    ifstream sub("msmarco_passages_subset.tsv");
    if (sub) {
        string id;
        while (getline(sub, id)) {
            id.erase(remove_if(id.begin(), id.end(), ::isspace), id.end());
            if (!id.empty()) allowed.insert(id);
        }
        cerr << "Subset: " << allowed.size() << " IDs\n";
    }

    vector<tuple<string, int, int>> buf;
    const size_t MAX_BUF = 10000000;
    int docID = 0, run = 0;

    ofstream pageTable("index/page_table.txt");
    ofstream docLen("index/doc_lengths.txt");
    ofstream docStore("index/documents.dat", ios::binary);
    ofstream docIdx("index/documents.idx", ios::binary);

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string pid, text;
        if (!getline(ss, pid, '\t') || !getline(ss, text)) continue;
        if (!allowed.empty() && !allowed.count(pid)) continue;

        long long off = docStore.tellp();
        int len = text.length();
        docStore.write(text.c_str(), len);
        docIdx.write((char*)&off, 8);
        docIdx.write((char*)&len, 4);

        auto tokens = tokenize(text);
        if (tokens.empty()) continue;

        pageTable << docID << "\t" << pid << "\n";
        docLen << docID << "\t" << tokens.size() << "\n";

        unordered_map<string, int> tf;
        for (auto& t : tokens) tf[t]++;
        for (auto& [term, freq] : tf)
            buf.emplace_back(term, docID, freq);

        docID++;
        if (docID % 100000 == 0)
            cerr << "Indexed " << docID << " docs\n";

        if (buf.size() >= MAX_BUF) {
            sort(buf.begin(), buf.end());
            writePartial(buf, run++);
            buf.clear();
        }
    }

    if (!buf.empty()) {
        sort(buf.begin(), buf.end());
        writePartial(buf, run++);
    }

    ofstream meta("index/indexer_meta.txt");
    meta << "total_documents\t" << docID << "\n";
    meta << "total_runs\t" << run << "\n";

    cerr << "Done: " << docID << " docs, " << run << " runs\n";
    return 0;
}
