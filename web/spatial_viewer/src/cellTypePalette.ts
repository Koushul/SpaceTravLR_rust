function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const m = l - c / 2;
  let rp = 0;
  let gp = 0;
  let bp = 0;
  if (hp >= 0 && hp < 1) {
    rp = c;
    gp = x;
  } else if (hp < 2) {
    rp = x;
    gp = c;
  } else if (hp < 3) {
    gp = c;
    bp = x;
  } else if (hp < 4) {
    gp = x;
    bp = c;
  } else if (hp < 5) {
    rp = x;
    bp = c;
  } else {
    rp = c;
    bp = x;
  }
  return [
    Math.round((rp + m) * 255),
    Math.round((gp + m) * 255),
    Math.round((bp + m) * 255),
  ];
}

export function rgbForCategoryIndex(
  idx: number,
  nCategories: number,
): [number, number, number] {
  if (nCategories <= 0) return [100, 100, 110];
  const golden = 0.618033988749895;
  const h = ((idx * golden) % 1) * 360;
  return hslToRgb(h, 0.72, 0.52);
}
