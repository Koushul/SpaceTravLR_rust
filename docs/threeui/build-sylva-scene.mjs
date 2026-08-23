import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const sourcePath = join(
  root,
  "src/shaders/sylva-living-world/sources/inner-green-3d.html"
);
const runtimePath = join(
  root,
  "src/shaders/sylva-living-world/sources/inner-green-assets/three.min.js"
);
const outputPath = join(root, "sylva-tree-scene.html");

const innerGreenSource = readFileSync(sourcePath, "utf8");
const threeRuntime = readFileSync(runtimePath, "utf8");

const presentationStart = innerGreenSource.indexOf('<main class="hero" id="hero">');
const runtimeStart = innerGreenSource.indexOf(
  '<script src="inner-green-assets/three.min.js"></script>'
);

if (presentationStart < 0 || runtimeStart < 0 || runtimeStart <= presentationStart) {
  throw new Error("Could not isolate the Sylva Three.js scene from inner-green-3d.html.");
}

const wordmarkLetters = "SpaceTravLR"
  .split("")
  .map((ch) => `<span>${ch}</span>`)
  .join("");

const sceneOnlyMarkup =
  `<main class="hero" id="hero"><canvas id="scene" role="img" aria-label="Sylva Living Green"></canvas><div class="stage" id="stage" aria-hidden="true"></div><div class="st-wordmark" aria-hidden="true">${wordmarkLetters}</div></main>`;

const sceneOnlyStyle = `<style data-threeui-sylva-scene>
@font-face{font-family:Nasalization;src:url("fonts/nasalization-rg.woff") format("woff");font-weight:400;font-style:normal;font-display:swap}
html{color-scheme:light}
html[data-scheme="slate"]{color-scheme:dark}
html,body{width:100%!important;height:100%!important;min-height:0!important;margin:0!important;overflow:hidden!important;background:transparent!important}
body{position:relative!important;background:transparent!important}
.hero{position:relative!important;width:100%!important;height:100%!important;min-height:0!important;overflow:hidden!important;background:transparent!important}
.hero::after{display:none!important}
.stage{position:absolute!important;inset:0!important;margin:0!important;width:100%!important;height:100%!important;background:transparent!important}
#scene{pointer-events:auto!important;background:transparent!important}
.st-wordmark{position:absolute;left:50%;top:48%;z-index:20;margin:0;padding:0;border:0;display:flex;gap:.04em;font-family:Nasalization,"Helvetica Neue",Helvetica,Arial,sans-serif;font-weight:400;font-size:clamp(38px,8.2vw,84px);letter-spacing:.08em;line-height:1;white-space:nowrap;user-select:none;pointer-events:none;color:rgba(48,54,42,.55);transform:translate(-50%,-50%);transition:top 3.4s cubic-bezier(.16,1,.3,1)}
.st-wordmark.is-risen{top:14%;animation:st-word-float 10s ease-in-out 3.4s infinite}
.st-wordmark span{display:inline-block;pointer-events:auto;cursor:default;transform:translateY(0) scale(1);transition:transform .28s cubic-bezier(.22,.61,.36,1),color .28s ease,text-shadow .28s ease}
.st-wordmark span:hover,.st-wordmark span.is-hot{transform:translateY(-16px) scale(1.16);color:rgba(40,48,36,.78);text-shadow:0 10px 22px rgba(72,80,66,.18)}
[data-scheme="slate"] .st-wordmark{color:rgba(236,232,248,.46)}
[data-scheme="slate"] .st-wordmark span:hover,[data-scheme="slate"] .st-wordmark span.is-hot{color:rgba(244,240,255,.86);text-shadow:0 10px 24px rgba(12,10,20,.35)}
@keyframes st-word-float{0%,100%{transform:translate(-50%,-50%) translateY(0)}50%{transform:translate(-50%,-50%) translateY(-10px)}}
@media (prefers-reduced-motion:reduce){.st-wordmark,.st-wordmark.is-risen{top:14%;transition:none;animation:none}.st-wordmark span,.st-wordmark span:hover{transition:none;transform:none}}
</style>`;

function mustReplace(haystack, search, replacement) {
  if (!haystack.includes(search)) {
    throw new Error(`Sylva scene adapter missing expected source snippet:\n${search.slice(0, 180)}`);
  }
  return haystack.replace(search, replacement);
}

let output = `${innerGreenSource.slice(0, presentationStart)}${sceneOnlyMarkup}

${innerGreenSource.slice(runtimeStart)}`;

output = mustReplace(output, "</head>", `${sceneOnlyStyle}</head>`);
output = mustReplace(
  output,
  '<script src="inner-green-assets/three.min.js"></script>',
  `<script>${threeRuntime}</script>`
);
output = mustReplace(
  output,
  "var scanning = false, scanT = 0, scanMax = 3000;",
  `var scanning = false, scanT = 0, scanMax = 3000, scanArmed = true;
  function beginScan() {
    if (!scanArmed || REDUCED || document.hidden || window.__sylvaVisible === false) return;
    scanArmed = false;
    uScanOn.value = 1;
    uScanR.value = 0;
    scanT = 0;
    scanning = true;
    var mark = document.querySelector(".st-wordmark");
    if (mark) mark.classList.add("is-risen");
  }`
);
output = mustReplace(
  output,
  "if (!REDUCED && !document.hidden) { uScanOn.value = 1; uScanR.value = 0; scanning = true; }",
  "beginScan();"
);
output = mustReplace(
  output,
  "var NARROW   = window.matchMedia('(max-width: 900px)');",
  "var NARROW   = { matches: false, addEventListener: function () {} };"
);
output = mustReplace(
  output,
  "renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: !small });",
  "renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: !small, powerPreference: 'high-performance', stencil: false });"
);
output = mustReplace(
  output,
  "if (!small) {\n      bf = buildButterfly(nearGroup, nearLimbs, nearGroup.userData.uni);\n      bee = buildHoneybee(nearGroup, nearLimbs, nearGroup.userData.uni);\n    }",
  "bf = buildButterfly(nearGroup, nearLimbs, nearGroup.userData.uni);\n    bee = buildHoneybee(nearGroup, nearLimbs, nearGroup.userData.uni);"
);
output = mustReplace(
  output,
  "(function loop() { requestAnimationFrame(loop); tick(); })();",
  `(function loop() {
      requestAnimationFrame(loop);
      if (document.hidden || window.__sylvaVisible === false) return;
      beginScan();
      tick();
    })();`
);
output = mustReplace(
  output,
  "if (renderer && clock) renderFrame();",
  "if (renderer && clock && !(REDUCED && frames > 1)) renderFrame();"
);
output = mustReplace(
  output,
  "startPortalReveal();\n    initDock();",
  ""
);
output = mustReplace(
  output,
  "src:url('inner-green-assets/lexend-latin.woff2') format('woff2');",
  "src:local('Lexend');"
);
output = mustReplace(output, "html{ background:#383b34; }", "html{ background:transparent; }");
output = mustReplace(output, "background:#383b34;", "background:transparent;");
output = mustReplace(
  output,
  "radial-gradient(70% 60% at 92% 8%,  rgba(24,28,20,.10) 0%, rgba(24,28,20,0) 68%),\n      #4a4d44;",
  "radial-gradient(70% 60% at 92% 8%,  rgba(24,28,20,.10) 0%, rgba(24,28,20,0) 68%),\n      transparent;"
);
output = mustReplace(
  output,
  "</body>",
  `<script>
(function () {
  if (window.parent !== window && window.__sylvaVisible === undefined) {
    window.__sylvaVisible = false;
  }
  var mark = document.querySelector(".st-wordmark");
  if (!mark) return;
  var letters = mark.querySelectorAll("span");
  function clear() {
    for (var i = 0; i < letters.length; i++) letters[i].classList.remove("is-hot");
  }
  window.addEventListener("pointermove", function (e) {
    var hot = document.elementFromPoint(e.clientX, e.clientY);
    clear();
    if (hot && hot.parentNode === mark) hot.classList.add("is-hot");
  }, { passive: true });
  window.addEventListener("pointerleave", clear);
})();
</script></body>`
);

writeFileSync(outputPath, output);
console.log(`wrote ${outputPath} (${output.length} bytes)`);
