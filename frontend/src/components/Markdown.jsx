// Minimal, dependency-free Markdown renderer for LLM answers: headings,
// bold/italic/code, bullet and numbered lists, horizontal rules, paragraphs.
// Renders React elements only (never innerHTML), so model output can't
// inject markup.

function inline(text, keyPrefix) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g;
  let last = 0;
  let m;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    const k = `${keyPrefix}-${i++}`;
    if (tok.startsWith("**")) out.push(<strong key={k} className="font-semibold text-ink">{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith("`")) out.push(<code key={k} className="rounded bg-surface-muted px-1 py-0.5 font-mono text-[0.85em]">{tok.slice(1, -1)}</code>);
    else out.push(<em key={k}>{tok.slice(1, -1)}</em>);
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function Markdown({ text }) {
  const lines = String(text || "").replace(/\r/g, "").split("\n");
  const blocks = [];
  let list = null; // { ordered, items }
  let para = [];

  const flushPara = () => {
    if (para.length) blocks.push({ type: "p", text: para.join(" ") });
    para = [];
  };
  const flushList = () => {
    if (list) blocks.push({ type: "list", ...list });
    list = null;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const t = line.trim();
    if (!t) { flushPara(); flushList(); continue; }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(t)) { flushPara(); flushList(); blocks.push({ type: "hr" }); continue; }
    const h = t.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flushPara(); flushList(); blocks.push({ type: "h", level: h[1].length, text: h[2] }); continue; }
    const ul = t.match(/^[-*•]\s+(.*)$/);
    const ol = t.match(/^\d{1,2}[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const ordered = Boolean(ol);
      if (!list || list.ordered !== ordered) { flushList(); list = { ordered, items: [] }; }
      list.items.push({ text: (ul || ol)[1], indent: line.length - line.trimStart().length > 1 });
      continue;
    }
    if (list && /^\s{2,}/.test(line)) { list.items[list.items.length - 1].text += ` ${t}`; continue; }
    flushList();
    para.push(t);
  }
  flushPara();
  flushList();

  return (
    <div className="space-y-2">
      {blocks.map((b, i) => {
        if (b.type === "hr") return <hr key={i} className="border-line" />;
        if (b.type === "h")
          return (
            <p key={i} className={`${b.level <= 2 ? "text-[15px]" : "text-sm"} pt-1 font-semibold text-ink`}>
              {inline(b.text, `h${i}`)}
            </p>
          );
        if (b.type === "list") {
          const Tag = b.ordered ? "ol" : "ul";
          return (
            <Tag key={i} className={`space-y-1 pl-5 ${b.ordered ? "list-decimal" : "list-disc"} marker:text-ink-3`}>
              {b.items.map((it, j) => (
                <li key={j} className={it.indent ? "ml-4" : ""}>{inline(it.text, `l${i}-${j}`)}</li>
              ))}
            </Tag>
          );
        }
        return <p key={i}>{inline(b.text, `p${i}`)}</p>;
      })}
    </div>
  );
}
