// K-way Merge with VarByte Compression
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>

using namespace std;

const int BLOCK_SIZE = 128;

struct Entry {
    string term;
    int doc, freq, file;

    bool operator>(const Entry& o) const {
        return term != o.term ? term > o.term : doc > o.doc;
    }
};

void vb_encode(int n, vector<unsigned char>& out) {
    while (n >= 128) {
        out.push_back((n & 0x7F) | 0x80);
        n >>= 7;
    }
    out.push_back(n & 0x7F);
}

void writeBlock(ofstream& f, const vector<int>& docs, const vector<int>& freqs,
                vector<int>& last, vector<int>& dsz, vector<int>& fsz) {
    // Delta encode
    vector<int> deltas;
    deltas.push_back(docs[0]);
    for (size_t i = 1; i < docs.size(); i++)
        deltas.push_back(docs[i] - docs[i-1]);

    // VarByte encode
    vector<unsigned char> ed, ef;
    for (int d : deltas) vb_encode(d, ed);
    for (int fr : freqs) vb_encode(fr, ef);

    int ds = ed.size(), fs = ef.size();
    f.write((char*)&ds, 4);
    f.write((char*)ed.data(), ds);
    f.write((char*)&fs, 4);
    f.write((char*)ef.data(), fs);

    last.push_back(docs.back());
    dsz.push_back(ds);
    fsz.push_back(fs);
}

bool readNext(ifstream& f, string& term, int& doc, int& freq) {
    int len;
    if (!f.read((char*)&len, 4)) return false;
    term.resize(len);
    f.read(&term[0], len);
    f.read((char*)&doc, 4);
    f.read((char*)&freq, 4);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <num_runs>\n";
        return 1;
    }

    int numRuns = stoi(argv[1]);

    vector<ifstream> runs(numRuns);
    for (int i = 0; i < numRuns; i++) {
        runs[i].open("partial/run_" + to_string(i) + ".bin", ios::binary);
        if (!runs[i]) {
            cerr << "Cannot open run " << i << "\n";
            return 1;
        }
    }

    ofstream inv("index/inverted_index.bin", ios::binary);
    ofstream lex("index/lexicon.txt");

    priority_queue<Entry, vector<Entry>, greater<Entry>> pq;

    vector<string> terms(numRuns);
    vector<int> docs(numRuns), freqs(numRuns);
    for (int i = 0; i < numRuns; i++) {
        if (readNext(runs[i], terms[i], docs[i], freqs[i]))
            pq.push({terms[i], docs[i], freqs[i], i});
    }

    vector<int> allLast, allDocSz, allFreqSz;
    unordered_map<string, int> df;
    string curTerm;
    vector<int> tDocs, tFreqs;
    long long startOff = 0;
    int startBlk = 0, nTerms = 0;
    long long np = 0;

    while (!pq.empty()) {
        auto e = pq.top();
        pq.pop();

        // New term - finish previous
        if (!curTerm.empty() && e.term != curTerm) {
            if (!tDocs.empty())
                writeBlock(inv, tDocs, tFreqs, allLast, allDocSz, allFreqSz);

            lex << curTerm << "\t" << startOff << "\t" << startBlk
                << "\t" << np << "\t" << df[curTerm] << "\n";

            tDocs.clear();
            tFreqs.clear();
            startOff = inv.tellp();
            startBlk = allLast.size();
            nTerms++;

            if (nTerms % 50000 == 0)
                cerr << "Merged " << nTerms << " terms\r" << flush;
        }

        if (e.term != curTerm) {
            curTerm = e.term;
            df[curTerm] = 0;
            np = 0;
        }

        tDocs.push_back(e.doc);
        tFreqs.push_back(e.freq);
        df[curTerm]++;
        np++;

        if ((int)tDocs.size() == BLOCK_SIZE) {
            writeBlock(inv, tDocs, tFreqs, allLast, allDocSz, allFreqSz);
            tDocs.clear();
            tFreqs.clear();
        }

        // Read next from same file
        int i = e.file;
        if (readNext(runs[i], terms[i], docs[i], freqs[i]))
            pq.push({terms[i], docs[i], freqs[i], i});
    }

    // Process last term
    if (!curTerm.empty()) {
        if (!tDocs.empty())
            writeBlock(inv, tDocs, tFreqs, allLast, allDocSz, allFreqSz);
        lex << curTerm << "\t" << startOff << "\t" << startBlk
            << "\t" << np << "\t" << df[curTerm] << "\n";
        nTerms++;
    }

    for (auto& f : runs) f.close();
    inv.close();
    lex.close();

    // Write metadata
    ofstream meta("index/metadata.bin", ios::binary);
    int nb = allLast.size();
    meta.write((char*)&nb, 4);
    meta.write((char*)allLast.data(), nb * 4);
    meta.write((char*)allDocSz.data(), nb * 4);
    meta.write((char*)allFreqSz.data(), nb * 4);

    ofstream stats("index/collection_stats.txt");
    stats << "total_terms\t" << nTerms << "\n";
    stats << "total_blocks\t" << nb << "\n";

    cerr << "\nDone: " << nTerms << " terms, " << nb << " blocks\n";
    return 0;
}
