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
  `<main class="hero is-ready" id="hero"><canvas id="scene" role="img" aria-label="Sylva Living Green"></canvas><div class="stage" id="stage" aria-hidden="true"></div><div class="st-wordmark" aria-hidden="true">${wordmarkLetters}</div></main>`;

const sceneOnlyStyle = `<style data-threeui-sylva-scene>
@font-face{font-family:Nasalization;src:url("fonts/nasalization-rg.woff") format("woff");font-weight:400;font-style:normal;font-display:swap}
html{color-scheme:light}
html[data-scheme="slate"]{color-scheme:dark}
html,body{width:100%!important;height:100%!important;min-height:0!important;margin:0!important;overflow:hidden!important;background:transparent!important}
body{position:relative!important;background:transparent!important}
.hero{position:relative!important;width:100%!important;height:100%!important;min-height:0!important;overflow:hidden!important;background:transparent!important;isolation:auto!important}
.hero::after{display:none!important}
.stage{position:absolute!important;inset:0!important;margin:0!important;width:100%!important;height:100%!important;background:transparent!important}
#scene{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;pointer-events:auto!important;opacity:1!important;z-index:1!important;background:transparent!important;display:block!important}
canvas#scene{background:transparent!important}
.st-wordmark{position:absolute;left:50%;top:14%;z-index:20;margin:0;padding:0;border:0;display:flex;gap:.04em;font-family:Nasalization,"Helvetica Neue",Helvetica,Arial,sans-serif;font-weight:400;font-size:clamp(38px,8.2vw,84px);letter-spacing:.08em;line-height:1;white-space:nowrap;user-select:none;pointer-events:none;color:rgba(48,54,42,.55);transform:translate(-50%,-50%);animation:st-word-float 10s ease-in-out infinite}
.st-wordmark span{display:inline-block;pointer-events:auto;cursor:default;transform:translateY(0) scale(1);transition:transform .28s cubic-bezier(.22,.61,.36,1),color .28s ease,text-shadow .28s ease}
.st-wordmark span:hover,.st-wordmark span.is-hot{transform:translateY(-16px) scale(1.16);color:rgba(40,48,36,.78);text-shadow:0 10px 22px rgba(72,80,66,.18)}
[data-scheme="slate"] .st-wordmark{color:rgba(236,232,248,.46)}
[data-scheme="slate"] .st-wordmark span:hover,[data-scheme="slate"] .st-wordmark span.is-hot{color:rgba(244,240,255,.86);text-shadow:0 10px 24px rgba(12,10,20,.35)}
@keyframes st-word-float{0%,100%{transform:translate(-50%,-50%) translateY(0)}50%{transform:translate(-50%,-50%) translateY(-10px)}}
@media (prefers-reduced-motion:reduce){.st-wordmark{animation:none}.st-wordmark span,.st-wordmark span:hover{transition:none;transform:none}}
</style>`;

let output = `${innerGreenSource.slice(0, presentationStart)}${sceneOnlyMarkup}

${innerGreenSource.slice(runtimeStart)}`
  .replace("</head>", `${sceneOnlyStyle}</head>`)
  .replace(
    '<script src="inner-green-assets/three.min.js"></script>',
    `<script>${threeRuntime}</script>`
  )
  .replace(
    "if (!REDUCED && !document.hidden) { uScanOn.value = 1; uScanR.value = 0; scanning = true; }",
    "uScanOn.value = 0; uScanR.value = scanMax; scanning = false;"
  )
  .replace(
    "setTimeout(function () { document.body.classList.add('intro-done'); }, REDUCED ? 0 : 2900);",
    "document.body.classList.add('intro-done');"
  )
  .replace(
    "var NARROW   = window.matchMedia('(max-width: 900px)');",
    "var NARROW   = { matches: false, addEventListener: function () {} };"
  )
  .replace(
    "var ARCH   = { w: 1900, left: -180, top: 306, aspect: 2800 / 1377 };",
    "var ARCH   = { w: 1480, left: 30, top: 280, aspect: 2800 / 1377 };"
  )
  .replace(
    "var FAR    = { w: 1150, left:  -40, top: 320, aspect: 1600 /  757, z: -260 };",
    "var FAR    = { w: 900, left:  85, top: 294, aspect: 1600 /  757, z: -260 };"
  )
  .replace(
    "uHaze:    { value: 0.14 },",
    "uHaze:    { value: 0.0 },"
  )
  .replace(
    "haze: 0.15, fog: 0.0, alpha: 1.0, order: 2,",
    "haze: 0.0, fog: 0.0, alpha: 1.0, order: 2,"
  )
  .replace(
    "haze: 0.16, fog: 0.26, alpha: 1.0, order: 0,",
    "haze: 0.0, fog: 0.0, alpha: 1.0, order: 0,"
  )
  .replace(
    "map: radialTexture(256, [[0, 'rgba(12,16,10,0.62)'], [0.45, 'rgba(12,16,10,0.26)'], [1, 'rgba(12,16,10,0)']]),",
    "map: radialTexture(256, [[0, 'rgba(12,16,10,0)'], [1, 'rgba(12,16,10,0)']]),"
  )
  .replace(
    "map: radialTexture(256, [[0, 'rgba(226,236,212,0.30)'], [0.42, 'rgba(214,226,200,0.10)'], [1, 'rgba(214,226,200,0)']]),",
    "map: radialTexture(256, [[0, 'rgba(226,236,212,0)'], [1, 'rgba(214,226,200,0)']]),"
  )
  .replace(
    "renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: !small });",
    "renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: false, premultipliedAlpha: true, powerPreference: 'high-performance', stencil: false });"
  )
  .replace(
    "renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, small ? 1.6 : 2));",
    "renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));"
  )
  .replace(
    "renderer.setClearColor(0x000000, 0);",
    "renderer.setClearColor(0x000000, 0); renderer.setClearAlpha(0); canvas.style.background = 'transparent';"
  )
  .replace(
    "if (!small) {\n      bf = buildButterfly(nearGroup, nearLimbs, nearGroup.userData.uni);\n      bee = buildHoneybee(nearGroup, nearLimbs, nearGroup.userData.uni);\n    }",
    "bf = buildButterfly(nearGroup, nearLimbs, nearGroup.userData.uni);\n    bee = buildHoneybee(nearGroup, nearLimbs, nearGroup.userData.uni);"
  )
  .replaceAll("wire: true,", "wire: false,")
  .replace("scene.add(shadowMesh);", "")
  .replace("scene.add(glowMesh);", "")
  .replace(
    "(function loop() { requestAnimationFrame(loop); tick(); })();",
    `(function loop() {
      requestAnimationFrame(loop);
      if (document.hidden || window.__sylvaVisible === false) return;
      tick();
    })();`
  )
  .replace(
    "if (renderer && clock) renderFrame();",
    "if (renderer && clock && !(REDUCED && frames > 1)) renderFrame();"
  )
  .replace(
    "startPortalReveal();\n    initDock();",
    ""
  )
  .replace(
    "src:url('inner-green-assets/lexend-latin.woff2') format('woff2');",
    "src:local('Lexend');"
  )
  .replace("html{ background:#383b34; }", "html{ background:transparent; }")
  .replace("background:#383b34;", "background:transparent;")
  .replaceAll("isolation:isolate;", "isolation:auto;")
  .replace(
      "radial-gradient(70% 60% at 92% 8%,  rgba(24,28,20,.10) 0%, rgba(24,28,20,0) 68%),\n      #4a4d44;",
      "radial-gradient(70% 60% at 92% 8%,  rgba(24,28,20,.10) 0%, rgba(24,28,20,0) 68%),\n      transparent;"
  )
  .replace(
    "</body>",
    `<script>
(function () {
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
