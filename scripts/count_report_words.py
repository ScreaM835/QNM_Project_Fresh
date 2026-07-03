"""Independent handbook word counter for paper/main.tex.

Counting rules (as established for the MPhil DIS handbook):
  COUNTED : body prose, section/subsection/paragraph headings, abstract,
            footnote text, appendix prose.
  FREE    : tables (entire float incl. cells/notes), figures (incl. tikz),
            captions, ALL math (inline $..$/\(..\), display \[..\],
            equation/align environments), \cite/\ref/\label tokens,
            bibliography, comments, preamble.
  REPORTED SEPARATELY (judgement call): algorithm floats.

Cross-check against: texcount -inc -sum main.tex  ("words in text" + "words in headers").
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")

WORD = re.compile(r"[A-Za-z]+(?:[-'\u2019][A-Za-z]+)*")

# environments removed wholesale (content is free)
FREE_ENVS = ["table", "figure", "tikzpicture", "equation", "align",
             "thebibliography", "tabular"]
# environments counted separately as a judgement-call category
REPORT_ENVS = ["algorithm"]
# commands removed together with their {argument}
DROP_CMDS = ["cite", "citep", "citet", "citealt", "ref", "eqref", "label",
             "includegraphics", "bibliography", "bibliographystyle",
             "usepackage", "newcommand", "renewcommand", "setlength",
             "vspace", "hspace", "title", "author", "date", "input"]
HEADER_CMDS = ["section", "subsection", "subsubsection", "paragraph",
               "subparagraph"]


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def find_braced(s, start):
    """Return (content, end_index) for the {...} group starting at s[start]=='{'."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif s[i] == "}" and s[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i
    return s[start + 1:], len(s)


def remove_envs(s, names):
    removed_words = 0
    for name in names:
        pat = re.compile(r"\\begin\{" + name + r"\*?\}.*?\\end\{" + name + r"\*?\}", re.S)
        while True:
            m = pat.search(s)
            if not m:
                break
            removed_words += len(WORD.findall(m.group(0)))
            s = s[:m.start()] + " " + s[m.end():]
    return s, removed_words


def remove_math(s):
    n = 0
    for pat in (r"\\\[.*?\\\]", r"\\\(.*?\\\)", r"\$\$.*?\$\$", r"\$[^$]*\$"):
        s, k = re.subn(pat, " ", s, flags=re.S)
        n += k
    return s, n


def extract_headers(s):
    """Remove heading commands, returning (text_without_headings, header_words)."""
    header_words = 0
    out = []
    i = 0
    pat = re.compile(r"\\(" + "|".join(HEADER_CMDS) + r")\*?\s*(\[[^\]]*\])?\s*\{")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        content, end = find_braced(s, m.end() - 1)
        header_words += len(WORD.findall(content))
        i = end + 1
    return "".join(out), header_words


def drop_commands(s):
    """Remove DROP_CMDS with their argument; unwrap all other \\cmd{...}."""
    pat = re.compile(r"\\([a-zA-Z]+)\*?\s*(\[[^\]]*\])?")
    out = []
    i = 0
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        j = m.end()
        if j < len(s) and s[j] == "{" and m.group(1) in DROP_CMDS:
            _, end = find_braced(s, j)
            i = end + 1
        else:
            i = j  # keep any following {...} content (unwrapped later)
    s = "".join(out)
    return s.replace("{", " ").replace("}", " ").replace("~", " ").replace("&", " ")


def count_file(src, is_main=False):
    s = strip_comments(src)
    if is_main:
        s = s.split(r"\begin{document}", 1)[1]
        s = s.split(r"\end{document}", 1)[0]
    s, table_fig_words = remove_envs(s, FREE_ENVS)
    s, algo_words = remove_envs(s, REPORT_ENVS)
    # captions outside floats (defensive)
    while True:
        m = re.search(r"\\caption\s*\{", s)
        if not m:
            break
        _, end = find_braced(s, m.end() - 1)
        s = s[:m.start()] + " " + s[end + 1:]
    s, _ = remove_math(s)
    s, header_words = extract_headers(s)
    s = drop_commands(s)
    text_words = len(WORD.findall(s))
    return text_words, header_words, table_fig_words, algo_words


def main():
    files = [("main.tex", True)]
    files += [(os.path.join("sections", f), False)
              for f in sorted(os.listdir(os.path.join(PAPER, "sections")))
              if f.endswith(".tex")]
    # only files actually \input by main.tex
    main_src = read(os.path.join(PAPER, "main.tex"))
    used = set(re.findall(r"\\input\{([^}]+)\}", main_src))
    files = [(f, m) for f, m in files
             if m or f.replace("\\", "/").removesuffix(".tex") in used]

    tot_t = tot_h = tot_free = tot_algo = 0
    print(f"{'file':<38}{'text':>7}{'headers':>9}{'free(tab/fig/math-env)':>24}{'algorithm':>11}")
    for fname, is_main in files:
        t, h, free, algo = count_file(read(os.path.join(PAPER, fname)), is_main)
        tot_t += t; tot_h += h; tot_free += free; tot_algo += algo
        print(f"{fname:<38}{t:>7}{h:>9}{free:>24}{algo:>11}")
    print("-" * 89)
    print(f"{'TOTAL':<38}{tot_t:>7}{tot_h:>9}{tot_free:>24}{tot_algo:>11}")
    print()
    print(f"HANDBOOK COUNT (text + headers)          : {tot_t + tot_h}")
    print(f"  ... if algorithm listings also counted : {tot_t + tot_h + tot_algo}")


if __name__ == "__main__":
    main()
