import importlib.util, sys
wd = sys.argv[1]
spec = importlib.util.spec_from_file_location("textutils", wd + "/textutils.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fails = []
def eq(name, got, want):
    if got != want: fails.append(f"{name}: got {got!r} want {want!r}")
eq("slugify1", m.slugify("Hello, World!"), "hello-world")
eq("slugify2", m.slugify("  --Already--Slug--  "), "already-slug")
eq("slugify3", m.slugify("Rust & Metal 2026"), "rust-metal-2026")
eq("truncate1", m.truncate("abcdefghij", 6), "abc...")
eq("truncate2", m.truncate("abc", 6), "abc")
eq("truncate3", m.truncate("abcdef", 6), "abcdef")
eq("truncate4", m.truncate("abcdefgh", 3), "...")
eq("truncate5", len(m.truncate("x" * 50, 10)), 10)
eq("wordfreq1", m.word_freq("The the THE cat"), [("the", 3), ("cat", 1)])
eq("wordfreq2", m.word_freq("b a B c a"), [("a", 2), ("b", 2), ("c", 1)])
eq("wrap", m.wrap("aa bb cc dd", 5), ["aa bb", "cc dd"])
eq("title", m.title_case("the lord of the rings"), "The Lord of the Rings")
print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
