# evaluate.py — production-parity check on rag_core.py (same metrics as notebook Section 10-11).

import re

import rag_core

# Ground truth: hand-labeled in Section 10.
TEST_SET = [
    ("What were Apple's total net sales for the three months ended December 30, 2023?", {1, 4}),
    ("What were Apple's iPhone net sales for the three months ended December 30, 2023?", {4}),
    ("Which region accounted for the highest net sales in the three months ended December 30, 2023?", {4}),
    ("What was Apple's net income for the three months ended December 30, 2023?", {3}),
    ("How did Apple's services revenue compare year over year in the three months ended December 30, 2023?", {1}),
    ("What were iPhone net sales in the quarter ended September 30, 2023?", {25}),
]

FAITHFUL_QUESTIONS = [
    "What were Apple's total net sales for the three months ended December 30, 2023?",
    "What were Apple's iPhone net sales for the three months ended December 30, 2023?",
    "How did Apple's services business perform in the three months ended December 30, 2023?",
]


def p3(ids, relevant):
    return len(set(ids[:3]) & relevant) / 3


def eval_precision(state):
    retrievers = {
        "Dense":   lambda q, s: [i for i, _ in rag_core._dense(s, q, 3)],
        "BM25":    lambda q, s: [i for i, _ in rag_core._bm25(s, q, 3)],
        "RRF":     lambda q, s: [i for i, _ in rag_core._merge(s, q, 3)],
        "Rerank":  lambda q, s: [i for i, _, _ in rag_core._rerank(
                                     s, q, [i for i, _ in rag_core._merge(s, q, rag_core.POOL)], 3)],

    }
    print("=" * 70)
    print("PRECISION@3 — production parity")
    for name, fn in retrievers.items():
        rows = []
        for question, relevant in TEST_SET:
            ids = fn(question, state)
            rows.append(p3(ids, relevant))
        print(f"  {name:8s} | " + "  ".join(f"{x:.2f}" for x in rows) +
              f"  | AVG {sum(rows)/len(rows):.2f}")


def eval_faithfulness(state):
    def figures(t):
        return {n.rstrip(",").rstrip(".") for n in re.findall(r"\d[\d,]*", t)}

    print("=" * 70)
    print("FAITHFULNESS — number-support check")
    for question in FAITHFUL_QUESTIONS:
        ranked = rag_core._rerank(state, question,
                                  [i for i, _ in rag_core._merge(state, question, rag_core.POOL)], 3)

        
        context = "\n".join(f"[{j}] {state['chunks'][cid].page_content}"
                            for j, (cid, _, _) in enumerate(ranked, start=1))
        result = rag_core.answer(question)
        unsupported = figures(result["answer"]) - figures(context)
        verdict = "FAITHFUL ✔" if not unsupported else f"UNFAITHFUL ✘ {sorted(unsupported)}"
        print(f"  {verdict:32s} | {result['answer'][:70]}")


if __name__ == "__main__":
    state = rag_core.get_state()
    eval_precision(state)
    eval_faithfulness(state)
    print("=" * 70)
    print("Done.")
