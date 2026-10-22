"""
Springer CCIS / LNCS submission format check for SOICT 2026.

There is no official automated checker for Springer proceedings the way ACL ships
aclpubcheck, so this encodes the rules from Springer's "Information for Authors of
Springer Computer Science Proceedings" and the SOICT call, and fails loudly on each.

Checks, in the order a desk reviewer would hit them:
  1. document class is llncs (not article/IEEEtran) and unmodified page geometry
  2. required front matter present: title, author, institute, abstract, keywords
  3. page budget: <= 12 pages EXCLUDING references
  4. every font embedded and no Type 3 bitmap fonts (Springer rejects these)
  5. no missing figures; all included graphics resolve
  6. no undefined references or citations in the build log
  7. no overfull boxes protruding into the margin beyond a tolerance
  8. PDF opens, page size is A4 or Letter and consistent across pages
  9. no placeholder text or unresolved template tokens left in the body

Usage:  python paper/format_check.py paper/essence_soict.tex
Exit code 0 = pass, 1 = at least one FAIL.
"""
from pathlib import Path
import re
import subprocess
import sys

TEX_PAGE_LIMIT_EXCL_REFS = 12
# SOICT 2026 camera-ready: short papers 8-11 pages INCLUDING figures, tables and
# references (full papers 12-15). Paper 4101's acceptance letter specifies 8-11.
CR_PAGE_MIN, CR_PAGE_MAX = 8, 11
OVERFULL_TOLERANCE_PT = 5.0     # boxes protruding less than this are cosmetic


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str))


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def main(tex_path):
    tex_path = Path(tex_path).resolve()
    d = tex_path.parent
    pdf = tex_path.with_suffix(".pdf")
    log = tex_path.with_suffix(".log")
    src = tex_path.read_text()
    results = []

    print(f"Springer CCIS format check — {tex_path.name}\n")

    # 1. document class / geometry
    cls = re.search(r"\\documentclass(\[[^\]]*\])?\{(\w+)\}", src)
    results.append(check("document class is llncs",
                         cls and cls.group(2) == "llncs",
                         f"found {cls.group(2) if cls else 'none'}"))
    bad_geom = re.findall(r"\\usepackage(\[[^\]]*\])?\{geometry\}|\\(textwidth|textheight|oddsidemargin|topmargin)\s*=",
                          src)
    results.append(check("page geometry not overridden", not bad_geom,
                         f"{len(bad_geom)} geometry override(s)" if bad_geom else "llncs defaults intact"))

    # 2. front matter
    for macro in ["title", "author", "institute", "keywords"]:
        results.append(check(rf"\{macro}{{...}} present", rf"\{macro}{{" in src))
    results.append(check(r"\begin{abstract} present", r"\begin{abstract}" in src))
    # llncs silently prints "Title Suppressed Due to Excessive Length" in the running
    # head when \titlerunning is missing and the title is long. It compiles cleanly, so
    # only reading the rendered page or checking here catches it.
    long_title = len(re.search(r"\\title\{(.+?)\}", src, re.S).group(1)) > 55 if \
        re.search(r"\\title\{(.+?)\}", src, re.S) else False
    results.append(check(r"\titlerunning{...} present (long title)",
                         (r"\titlerunning{" in src) or not long_title,
                         "required: llncs substitutes a placeholder otherwise"
                         if long_title and r"\titlerunning{" not in src else ""))

    # 3. page budget
    if not pdf.exists():
        results.append(check("compiled PDF exists", False, "run pdflatex first"))
        return finish(results)
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(pdf))
        n = len(doc)
        texts = [doc[i].get_textpage().get_text_range() for i in range(n)]
        ref_page = next((i + 1 for i, t in enumerate(texts)
                         if t.lstrip().startswith("References") or "\nReferences" in t), n + 1)
        # ref_page - 1 is WRONG when the references heading sits partway down a page:
        # the body text above it occupies that page too, so the body really runs onto it.
        # A reviewer counting pages counts that page against the limit.
        spill = 0
        if ref_page <= n:
            above = texts[ref_page - 1].split("References")[0]
            spill = len(above.split())
        body = ref_page - 1 + (1 if spill > 25 else 0)
        # CAMERA-READY rule differs from the submission rule. soict.org/submission/
        # camera-ready-submission: full papers 12-15 pages, short papers 8-11 pages,
        # both "including figures and tables and references". The submission limit of 12
        # EXCLUDED references, so the body-pages figure below is informational only --
        # the binding number for the camera-ready is the TOTAL page count.
        detail = (f"{n} total pages (camera-ready limit counts references); "
                  f"body {body}, references start p{ref_page}")
        results.append(check(f"camera-ready page budget {CR_PAGE_MIN}-{CR_PAGE_MAX} incl. references",
                             CR_PAGE_MIN <= n <= CR_PAGE_MAX, detail))
        results.append(check(f"submission-era budget <= {TEX_PAGE_LIMIT_EXCL_REFS} excl. references",
                             body <= TEX_PAGE_LIMIT_EXCL_REFS, f"{body} body pages"))
        sizes = {(round(doc[i].get_width()), round(doc[i].get_height())) for i in range(n)}
        results.append(check("page size consistent", len(sizes) == 1, f"{sizes}"))
        # Springer's own llncs template compiles to A4, and llncs fixes the text
        # block independently of paper size -- so A4 vs Letter changes only the
        # surrounding margin, never the line breaking (verified: the two PDFs'
        # text streams are byte-identical). Letter is what a US TeX install gives
        # you by default and is easy to ship by accident, so this is asserted
        # rather than left to chance.
        A4 = (595, 842)
        results.append(check("page size is A4 (Springer template default)",
                             sizes == {A4},
                             f"{sorted(sizes)[0]} -- add a4paper to \\documentclass"
                             if sizes != {A4} else "595x842pt"))
        # 9. placeholders
        flat = re.sub(r"\s+", " ", " ".join(texts))
        toks = re.findall(r"\b(TODO|TBD|XXX|FIXME|CITE|PLACEHOLDER|Lorem ipsum)\b|@@\w+@@|\{\{\w+\}\}"
                          r"|Title Suppressed Due to Excessive Length"
                          r"|Author Suppressed Due to Excessive Length",
                          flat, re.I)
        results.append(check("no placeholder text or unresolved tokens", not toks,
                             f"found {sorted(set(t for t in toks if t))[:4]}" if toks else ""))
    except ImportError:
        results.append(check("pypdfium2 available for PDF inspection", False))

    # 4. fonts embedded / no Type 3 — Springer rejects both, so this is checked
    #    through pikepdf rather than the pdffonts binary, which is not always present.
    try:
        import pikepdf
        with pikepdf.open(str(pdf)) as p:
            fonts, not_emb, type3 = {}, [], []
            for page in p.pages:
                res = page.get("/Resources", {})
                for _, fnt in dict(res.get("/Font", {})).items():
                    base = str(fnt.get("/BaseFont", "?")).lstrip("/")
                    sub = str(fnt.get("/Subtype", ""))
                    fonts[base] = sub
                    if sub == "/Type3":
                        type3.append(base)
                        continue
                    desc = fnt.get("/FontDescriptor")
                    if sub == "/Type0":                      # composite: check descendants
                        for df in list(fnt.get("/DescendantFonts", [])):
                            desc = df.get("/FontDescriptor") or desc
                    embedded = bool(desc) and any(
                        k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3"))
                    if not embedded:
                        not_emb.append(base)
            not_emb = sorted(set(not_emb)); type3 = sorted(set(type3))
            results.append(check("all fonts embedded", not not_emb,
                                 f"{len(fonts)} fonts, all embedded" if not not_emb
                                 else f"NOT embedded: {not_emb[:4]}"))
            results.append(check("no Type 3 bitmap fonts", not type3,
                                 f"Type 3: {type3[:4]}" if type3 else ""))
    except ImportError:
        print("  [SKIP] font check — pikepdf not installed")

    # 5-7. build log
    if log.exists():
        lg = log.read_text(errors="replace")
        missing = re.findall(r"File `([^']+)' not found", lg)
        results.append(check("no missing include files", not missing, f"{missing[:3]}" if missing else ""))
        undef_ref = re.findall(r"LaTeX Warning: Reference `([^']+)' on page", lg)
        undef_cit = re.findall(r"LaTeX Warning: Citation `([^']+)'", lg)
        results.append(check("no undefined references", not undef_ref, f"{undef_ref[:3]}" if undef_ref else ""))
        results.append(check("no undefined citations", not undef_cit, f"{undef_cit[:3]}" if undef_cit else ""))
        bad_of = [float(m) for m in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", lg)
                  if float(m) > OVERFULL_TOLERANCE_PT]
        results.append(check(f"no overfull boxes > {OVERFULL_TOLERANCE_PT}pt", not bad_of,
                             f"{len(bad_of)} boxes, worst {max(bad_of):.1f}pt" if bad_of else
                             "within tolerance"))
    else:
        print("  [SKIP] log checks — no .log file")

    # graphics resolve
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
    miss = [f for f in figs if not any((d / (f + e)).exists() for e in ["", ".pdf", ".png", ".jpg", ".eps"])]
    results.append(check("all figures resolve on disk", not miss,
                         f"{len(figs)} figures" if not miss else f"missing: {miss}"))

    return finish(results)


def finish(results):
    n_fail = sum(1 for r in results if not r)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        print(f"{n_fail} FAILURE(S) — fix before submitting")
    else:
        print("All checks passed — no desk-reject risk from format")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "paper/essence_soict.tex"))
