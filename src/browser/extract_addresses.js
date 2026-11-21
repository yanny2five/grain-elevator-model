// Extract US postal address blocks from a company "locations" page.
//
// Many co-op / grain-company sites block scripted downloads (Cloudflare,
// HTTP 403/429), so pages are opened in a normal browser and this function is
// run in the page (browser console, or the agent's browser tool). The JSON it
// returns is saved to data/addresses/company_pages/<slug>.json and parsed by
// src/company_locations.py. Provenance (URL, title, capture time) is included.
//
// Returns: {url, title, captured_at, n, records: [{street, city, state, zip,
//           label, heading, context}]}
(() => {
  const root = document.querySelector("main") || document.body;
  const lines = root.innerText.split("\n").map((s) => s.trim()).filter(Boolean);
  const CSZ = /^(.*?)[,\s]+([A-Z]{2})\.?\s+(\d{5})(?:-\d{4})?$/;              // "Akron, CO 80720"
  const ONE = /^(\d[^,]{2,60}?),\s*([A-Za-z .'-]{2,40}),\s*([A-Z]{2})\.?\s+(\d{5})/; // "725 Main St, Akron, CO 80720"
  const POBOX = /^(p\.?\s*o\.?\s*box|po box|box \d)/i;
  const isLabel = (s) => /:$/.test(s) && s.length < 60;
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const L = lines[i];
    let rec = null;
    let m = L.match(ONE);
    if (m) {
      rec = { street: m[1], city: m[2].trim(), state: m[3], zip: m[4], at: i };
    } else if ((m = L.match(CSZ)) && m[1] && m[1].length < 40 && !/\d{3}/.test(m[1])) {
      let street = "";
      for (let j = i - 1; j >= Math.max(0, i - 3); j--) {
        const s = lines[j];
        if (POBOX.test(s)) continue;
        if (isLabel(s)) break;
        if (/\d|hwy|highway|road|rd\b|street|st\b|ave|county|cr\b|route/i.test(s) && s.length < 80) { street = s; break; }
      }
      rec = { street, city: m[1].replace(/,$/, "").trim(), state: m[2], zip: m[3], at: i };
    }
    if (!rec) continue;
    let label = "";
    for (let j = rec.at - 1; j >= Math.max(0, rec.at - 4); j--) if (isLabel(lines[j])) { label = lines[j].replace(/:$/, ""); break; }
    // Nearest preceding short line that looks like a location heading.
    let heading = "";
    for (let j = rec.at - 1; j >= Math.max(0, rec.at - 25); j--) {
      const s = lines[j];
      if (!isLabel(s) && s.length < 60 && !/\d{3}/.test(s) && /^[A-Z]/.test(s) && !/^(phone|fax|hours|services|address)/i.test(s)) { heading = s; }
      if (j < rec.at - 2 && heading && /,\s*[A-Z][a-z]+/.test(s)) { heading = s; break; }
    }
    out.push({ street: rec.street, city: rec.city, state: rec.state, zip: rec.zip, label, heading,
               context: lines.slice(Math.max(0, rec.at - 6), rec.at + 1).join(" | ").slice(0, 300) });
  }
  return { url: location.href, title: document.title, captured_at: new Date().toISOString(), n: out.length, records: out };
})();
