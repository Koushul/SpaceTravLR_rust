const ICONS = {
  run: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><polygon points="11 9 16 12 11 15" /></svg>`,
  examples: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6.75 7 10.25v7.75c0 1.15 1.45 1.85 2.45 1.35L12 17.25"/><path d="M12 6.75 17 10.25v7.75c0 1.15-1.45 1.85-2.45 1.35L12 17.25"/><path d="M12 6.75v10.5"/></svg>`,
  microniche: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="4"/></svg>`,
  perturb: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2 4 14h7l-1 8 9-12h-7l1-8z"/></svg>`,
  receptor: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="8" cy="11" r="2.5"/><circle cx="16" cy="11" r="2.5"/><path d="M10.5 11h3"/><path d="M12 11v4"/><circle cx="12" cy="17" r="1.75"/></svg>`,
  compare: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="9.75" cy="12" r="5.25"/><circle cx="14.25" cy="12" r="5.25"/></svg>`,
  network: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="8" r="2"/><circle cx="18" cy="7" r="2"/><circle cx="17" cy="17" r="2"/><circle cx="7" cy="16" r="2"/><path d="M7.5 9.5 16.5 8"/><path d="M8 14.5 15.5 15.5"/><path d="M7.8 9.8 7.2 14.2"/></svg>`,
  downloadDataset: `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="16" width="16" height="4" rx="1.5"/><path d="M12 4v10"/><path d="M8.5 12.5 12 16l3.5-3.5"/></svg>`,
  framework: `<svg viewBox="0 0 24 24" aria-hidden="true" class="st-i-want-to__glyph"><text x="12" y="12.5" text-anchor="middle" dominant-baseline="central" font-size="20" font-family="'Cambria Math', 'Latin Modern Math', 'STIX Two Math', serif" fill="currentColor">∫</text></svg>`,
};

const CARDS = [
  {
    title: "run SpaceTravLR on my own data",
    description: "Install and run SpaceTravLR on your data. Requires a GPU.",
    href: "install/#quick-install-recommended",
    icon: ICONS.run,
  },
  {
    title: "look at some examples",
    description: "Some useful downstream analysis",
    href: "examples/atera-cervical-cancer-human/",
    icon: ICONS.examples,
  },
  {
    title: "understand the method",
    description:
      "Show me the math & code",
    href: "math/",
    icon: ICONS.framework,
  },
  {
    title: "find functional ligand–receptor interactions",
    description:
      "Prioritize ligand–receptor pairs in spatial context for follow-up validation.",
    href: "tutorials/screen-ligand-receptors/",
    icon: ICONS.receptor,
  },
  {
    title: "knockout a gene",
    description:
      "Simulate the cell intrinsic and extrinsic effects of perturbing a gene",
    href: "tutorials/perturb-genes-spatially/",
    icon: ICONS.perturb,
  },

  {
    title: "compare two samples",
    description:
      "Compare functional gene activities between two or more samples or conditions.",
    href: "tutorials/compare-two-samples/",
    icon: ICONS.compare,
  },
  {
    title: "find functional microniches",
    description:
      "",
    href: "tutorials/find-functional-microniches/",
    icon: ICONS.microniche,
  },
  {
    title: "visualise a spatial regulatory network",
    description:
      "Explore how perturbations propagate through spatially resolved gene regulatory and signalling networks.",
    href: "math/",
    icon: ICONS.network,
  },
  {
    title: "download a spatial dataset",
    description:
      "Browse example spatial transcriptomics datasets with public download links and sample workflows.",
    href: "examples/atera-cervical-cancer-human/",
    icon: ICONS.downloadDataset,
  },
];

function cardHtml(card, index) {
  return `
    <a
      class="st-i-want-to__card"
      href="${card.href}"
      style="--st-iw-i: ${index}"
    >
      <span class="st-i-want-to__icon">${card.icon}</span>
      <h3 class="st-i-want-to__title">${card.title}</h3>
      <p class="st-i-want-to__desc">${card.description}</p>
    </a>
  `;
}

function mountIWantToCards(root) {
  if (root.querySelector(".st-i-want-to__grid")) return;
  root.innerHTML = `<div class="st-i-want-to__grid">${CARDS.map(cardHtml).join("")}</div>`;
}

function initIWantToCards() {
  const root = document.getElementById("st-i-want-to");
  if (root) mountIWantToCards(root);
}

document$.subscribe(initIWantToCards);
