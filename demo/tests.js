import { _internal, correlationFromFamilies, deff, evaluate, mean, moments, rankCorr, simulate, solve, dot,
         walkForward, worldFromArrays } from "./engine.js";

const results = [];
function check(name, diff, tol) {
  const ok = Number.isFinite(diff) && diff <= tol;
  results.push({ name, diff, ok });
  const tr = document.createElement("tr");
  for (const [txt, cls] of [[name, ""], [Number.isFinite(diff) ? diff.toExponential(2) : String(diff), ""],
                            [ok ? "pass" : "FAIL", ok ? "pass" : "fail"]]) {
    const td = document.createElement("td"); td.textContent = txt; if (cls) td.className = cls; tr.appendChild(td);
  }
  document.getElementById("rows").appendChild(tr);
}
const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return a.length === b.length ? m : NaN; };
const scale = (a) => Math.max(1e-12, ...Array.from(a, Math.abs));

async function run() {
  const fx = await (await fetch("./fixtures/parity.json")).json();
  const W = worldFromArrays(fx);
  const mom = moments(W);
  const train = fx.train;
  check("design effect (deff)", Math.abs(deff(mom, train) - fx.deff), 1e-8 * fx.deff);
  for (const [name, ref] of Object.entries(fx.fit)) {
    const m = _internal.fitModel(name, W, mom, train, ref.params, {});
    if (name === "REGIME") {
      for (const r of ["0", "1"]) {
        const b = m.f.byRegime[Number(r)];
        check(`REGIME weights, regime ${r} (relative)`, maxAbs(b, ref.byRegime[r]) / scale(ref.byRegime[r]), 1e-6);
      }
      continue;
    }
    const b = m.beta();
    check(`${name} weights (relative)`, maxAbs(b, ref.beta) / scale(ref.beta), 1e-6);
  }
  for (const [name, ref] of Object.entries(fx.wf)) {
    const res = walkForward(W, mom, name, { start: 84, refit: 12, tuneEvery: 60, autoTune: false,
                                            manual: { [name]: ref.params } });
    const pred = res.pred.subarray(84 * W.N, 99 * W.N);
    check(`${name} walk-forward forecasts (relative)`, maxAbs(pred, ref.pred) / scale(ref.pred), 1e-6);
  }
  // simulation: standalone ICs and the oracle IC come out as configured
  const signals = [{ name: "A", family: "f1", ic: 0.05, persistence: 0.5 }, { name: "B", family: "f1", ic: 0.03, persistence: 0.5 },
                   { name: "C", family: "f2", ic: 0.04, persistence: 0.5 }, { name: "D", family: "f2", ic: 0.0, persistence: 0.5 }];
  const C = correlationFromFamilies(signals, 0.6, 0.1);
  const Wsim = simulate({ N: 400, T: 160, signals, corr: C, seed: 9, factorShare: 0 });
  for (let k = 0; k < 4; k++) {
    const ics = [];
    for (let t = 0; t < Wsim.T; t++) {
      const x = new Float64Array(Wsim.N); for (let i = 0; i < Wsim.N; i++) x[i] = Wsim.X[(t * Wsim.N + i) * 4 + k];
      ics.push(rankCorr(x, Wsim.ret.subarray(t * Wsim.N, (t + 1) * Wsim.N)));
    }
    check(`simulated standalone IC of ${signals[k].name} (target ${signals[k].ic})`, Math.abs(mean(ics) - signals[k].ic), 0.012);
  }
  const ic = Float64Array.from(signals.map((s) => s.ic)), target = Math.sqrt(dot(ic, solve(C, ic)));
  const ev = evaluate(Wsim, { pred: Wsim.oracle }, { start: 0 });
  check(`simulated oracle IC (target ${target.toFixed(3)})`, Math.abs(mean(ev.ic) - target), 0.012);
  const failed = results.filter((r) => !r.ok).length;
  document.getElementById("summary").textContent = `${results.length - failed} of ${results.length} checks passed.`;
  window.__parity = { passed: results.length - failed, total: results.length, results };
}
run().catch((e) => { document.getElementById("summary").textContent = "Error: " + e; window.__parity = { error: String(e) }; });
