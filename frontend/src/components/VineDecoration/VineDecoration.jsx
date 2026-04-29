import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import monsteraLeafUrl from './monsteraLeaf.svg';
import mediumLeafUrl from './mediumMonsteraLeaf.svg';
import './VineDecoration.css';

/* Preload SVG images — resolves when both are cached */
const svgReady = Promise.all(
  [monsteraLeafUrl, mediumLeafUrl].map(
    url => new Promise(resolve => { const img = new Image(); img.onload = resolve; img.onerror = resolve; img.src = url; })
  )
);

function useSvgReady() {
  const [ready, setReady] = useState(false);
  useEffect(() => { svgReady.then(() => setReady(true)); }, []);
  return ready;
}

/*
 * Complete vine + flower decoration system.
 * Vines hug the scanner edges: stems grow from bottom corners upward
 * along the scanner border, curving inward slightly near the top.
 * Flowers are integrated directly into the vine composition.
 *
 * Viewbox = 120 x 500. The scanner occupies roughly x:30–90, y:20–480.
 * Stems run along x≈20 (left) and x≈100 (right).
 */

/* ── Unique-ID generator for SVG gradients ── */
let _uid = 0;
const uid = () => `vd${++_uid}`;

/* ── Cohesive color palette ── */
const VINE = { stem: '#2a6e34', stemLight: '#4a9a4a', dark: '#1e5a28', mid: '#3ba55c', light: '#5cc66e', bright: '#6dd67c' };
const FLOWER = { coral: '#f2718a', yellow: '#f5c842', lavender: '#b497d6', blue: '#7ec8e3' };
const BG = 'rgba(168,200,232,0.65)';



/* ── MonsteraLeaf — detailed botanical shape with veins (from SVG) ── */
function MonsteraLeaf({ size = 1, angle = 0 }) {
  return (
    <g transform={`rotate(${angle}) scale(${size})`}>
      <g transform="scale(0.55) translate(-355,-230)">
        <image href={monsteraLeafUrl} x="270" y="30" width="180" height="210" />
      </g>
      <path d="M0 0 L0 10" stroke="#2D5A34" strokeWidth="2.5" strokeLinecap="round" opacity="0.45" fill="none" />
    </g>
  );
}



/* ── MediumMonsteraLeaf — botanical tropical leaf with detailed veins (from SVG) ── */
function MediumMonsteraLeaf({ angle = 0, size = 1 }) {
  return (
    <g transform={`rotate(${angle}) scale(${size})`}>
      <g transform="scale(0.35, -0.35) translate(-380, -470)">
        <image href={mediumLeafUrl} x="260" y="260" width="220" height="240" />
      </g>
      <path d="M0 0 L0 8" stroke="#2D5A34" strokeWidth="2" strokeLinecap="round" opacity="0.45" fill="none" />
    </g>
  );
}

/* ── MediumLeaf — simple elongated leaf, bigger than SmallLeaf ── */
function MediumLeaf({ angle = 0, color = '#3ba55c', size = 1 }) {
  return (
    <g transform={`rotate(${angle}) scale(${size})`}>
      <path
        d="M0 0 C-5 -6 -10 -18 -7 -30 C-4 -38 0 -42 0 -42 C0 -42 4 -38 7 -30 C10 -18 5 -6 0 0Z"
        fill={color} stroke={VINE.dark} strokeWidth="0.6"
      />
      <path d="M0 -3 L0 -40" stroke={VINE.dark} strokeWidth="0.6" strokeLinecap="round" opacity="0.3" fill="none" />
      <path d="M0 -12 C-3 -15 -5 -18 -5 -20" stroke={VINE.dark} strokeWidth="0.4" strokeLinecap="round" opacity="0.2" fill="none" />
      <path d="M0 -20 C3 -23 5 -26 5 -28" stroke={VINE.dark} strokeWidth="0.4" strokeLinecap="round" opacity="0.2" fill="none" />
      <path d="M0 -28 C-2 -30 -4 -33 -4 -34" stroke={VINE.dark} strokeWidth="0.4" strokeLinecap="round" opacity="0.2" fill="none" />
    </g>
  );
}

/* ── SmallLeaf — tiny accent ── */
function SmallLeaf({ angle = 0, color = '#5cb85c', size = 1 }) {
  return (
    <g transform={`rotate(${angle}) scale(${size})`}>
      <path
        d="M0 0 C-3 -4 -6 -11 -4 -18 C-2 -22 0 -24 0 -24 C0 -24 2 -22 4 -18 C6 -11 3 -4 0 0Z"
        fill={color} stroke={VINE.dark} strokeWidth="0.5"
      />
      <path d="M0 -2 L0 -22" stroke={VINE.dark} strokeWidth="0.5" strokeLinecap="round" opacity="0.3" fill="none" />
    </g>
  );
}

/* ── CartoonFlower — two variants: 4-petal and 8-petal ── */
function CartoonFlower({ color = FLOWER.coral, size = 1, angle = 0, petals = 4 }) {
  const id = uid();
  const four = [
    'M0 -5 C-7 -10 -8 -19 0 -24 C8 -19 7 -10 0 -5',        // top
    'M5 0 C10 -7 19 -8 24 0 C19 8 10 7 5 0',                 // right
    'M0 5 C7 10 8 19 0 24 C-8 19 -7 10 0 5',                  // bottom
    'M-5 0 C-10 7 -19 8 -24 0 C-19 -8 -10 -7 -5 0',          // left
  ];
  const eight = [
    'M0 -5 C-5 -10 -5 -18 0 -22 C5 -18 5 -10 0 -5',          // N
    'M3.5 -3.5 C7 -9 15 -13 19 -9 C16 -4 9 -1 3.5 -3.5',    // NE
    'M5 0 C10 -5 18 -5 22 0 C18 5 10 5 5 0',                  // E
    'M3.5 3.5 C9 7 13 15 9 19 C4 16 1 9 3.5 3.5',            // SE
    'M0 5 C5 10 5 18 0 22 C-5 18 -5 10 0 5',                  // S
    'M-3.5 3.5 C-7 9 -15 13 -19 9 C-16 4 -9 1 -3.5 3.5',    // SW
    'M-5 0 C-10 5 -18 5 -22 0 C-18 -5 -10 -5 -5 0',          // W
    'M-3.5 -3.5 C-9 -7 -13 -15 -9 -19 C-4 -16 -1 -9 -3.5 -3.5', // NW
  ];
  const paths = petals === 8 ? eight : four;
  return (
    <g transform={`rotate(${angle}) scale(${size})`}>
      <defs>
        <radialGradient id={`${id}f`} cx="40%" cy="40%" r="60%">
          <stop offset="0%" stopColor={color} stopOpacity="1" />
          <stop offset="100%" stopColor={color} stopOpacity="1" />
        </radialGradient>
      </defs>
      {paths.map((d, i) => (
        <path key={i} d={d} fill={`url(#${id}f)`} stroke={VINE.dark} strokeWidth="0.7" />
      ))}
      <circle cx="0" cy="0" r="4.5" fill="#fde68a" stroke="#e5b72e" strokeWidth="0.6" />
      <circle cx="0" cy="0" r="2.5" fill="#f5c842" />
      <circle cx="-1" cy="-1.5" r="0.8" fill="rgba(255,255,255,0.45)" />
    </g>
  );
}

/* ── AnimatedFlower — position wrapper + spring-animated inner ── */
function AnimatedFlower({ x, y, delay = 0, color, size = 1, angle = 0, petals = 4, visible = true }) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <motion.g
        initial={{ scale: 0, opacity: 0 }}
        animate={visible ? { scale: 1, opacity: 1 } : { scale: 0, opacity: 0 }}
        transition={{ delay: visible ? delay : 0.05, type: 'spring', stiffness: 180, damping: 14 }}
      >
        <CartoonFlower color={color} size={size} angle={angle} petals={petals} />
      </motion.g>
    </g>
  );
}

/* ── Tendril — small curling wisp ── */
function Tendril({ d }) {
  return <path d={d} stroke={VINE.stemLight} strokeWidth="1.2" strokeLinecap="round" fill="none" opacity="0.45" />;
}

/*
 * ══════════════════════════════════════════
 *   LEFT  VINE  (BEHIND)
 *   Stem hugs left scanner edge, bottom→top
 * ══════════════════════════════════════════
 */
function LeftVineBehind() {
  return (
    <svg className="vine-svg" viewBox="0 0 120 500" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Main stem — follows left scanner edge */}
      <path
        d="M60 500 C55 470 32 430 26 380 Q20 320 22 260 Q24 200 28 150 Q32 100 40 60 Q48 30 58 10"
        stroke={VINE.stem} strokeWidth="5.5" strokeLinecap="round" fill="none"
      />
      <path
        d="M58 498 C53 468 30 428 24 378 Q18 318 20 258 Q22 198 26 148 Q30 98 38 58 Q46 28 56 8"
        stroke={VINE.stemLight} strokeWidth="3" strokeLinecap="round" fill="none" opacity="0.3"
      />
      {/* A few leaves tucked behind scanner top */}
      <g transform="translate(46, 50)"><SmallLeaf angle={-30} color={VINE.bright} /></g>
      <g transform="translate(38, 80)"><MediumLeaf angle={-40} color={VINE.mid} size={0.8} /></g>
      <g transform="translate(30, 120)"><MediumMonsteraLeaf angle={25} size={0.6} /></g>
      {/* Leaves behind scanner bottom */}
      <g transform="translate(28, 400)"><SmallLeaf angle={-20} color={VINE.bright} /></g>
      <g transform="translate(35, 430)"><MediumLeaf angle={15} color={VINE.mid} size={0.7} /></g>
      <g transform="translate(48, 460)"><MediumMonsteraLeaf angle={-35} size={0.55} /></g>
    </svg>
  );
}

/*
 * ══════════════════════════════════════════
 *   RIGHT  VINE  (BEHIND)
 *   Stem hugs right scanner edge, bottom→top
 * ══════════════════════════════════════════
 */
function RightVineBehind() {
  return (
    <svg className="vine-svg" viewBox="0 0 120 500" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M60 500 C65 465 88 425 94 375 Q100 315 98 255 Q96 195 92 145 Q88 95 80 55 Q72 25 62 8"
        stroke={VINE.stem} strokeWidth="5.5" strokeLinecap="round" fill="none"
      />
      <path
        d="M62 498 C67 463 90 423 96 373 Q102 313 100 253 Q98 193 94 143 Q90 93 82 53 Q74 23 64 6"
        stroke={VINE.stemLight} strokeWidth="3" strokeLinecap="round" fill="none" opacity="0.3"
      />
      <g transform="translate(74, 42)"><SmallLeaf angle={28} color={VINE.bright} /></g>
      <g transform="translate(84, 75)"><MediumLeaf angle={38} color={VINE.mid} size={0.75} /></g>
      <g transform="translate(90, 115)"><MediumMonsteraLeaf angle={-20} size={0.6} /></g>
      <g transform="translate(92, 395)"><SmallLeaf angle={22} color={VINE.bright} /></g>
      <g transform="translate(84, 435)"><MediumLeaf angle={-18} color={VINE.mid} size={0.7} /></g>
      <g transform="translate(70, 465)"><MediumMonsteraLeaf angle={30} size={0.55} /></g>
    </svg>
  );
}

/*
 * ══════════════════════════════════════════
 *   LEFT  VINE  (FRONT)
 *   Main decorative layer — monsteras, medium leaves, flowers, tendrils.
 *   Stem runs close to the scanner left edge.
 * ══════════════════════════════════════════
 */
function LeftVineFront({ showFlowers }) {
  return (
    <svg className="vine-svg" viewBox="0 0 120 500" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Front stem segment */}
      <path
        d="M52 480 Q30 440 24 380 Q20 320 22 260 Q24 200 28 155 Q34 105 44 60"
        stroke={VINE.stem} strokeWidth="4.5" strokeLinecap="round" fill="none"
      />
      {/* Tendrils curling off the stem */}
      <Tendril d="M24 340 Q14 332 10 318 Q8 310 12 306" />
      <Tendril d="M28 195 Q16 188 12 174 Q10 168 15 162" />
      <Tendril d="M40 75 Q30 68 26 56 Q24 50 28 46" />

      {/* ── Leaf mix: small → medium → medium monstera → monstera ── */}
      {/* Small leaves (accents along stem) */}
      <g transform="translate(42, 72)"><SmallLeaf angle={-32} color={VINE.bright} /></g>
      <g transform="translate(24, 230)"><SmallLeaf angle={-18} color={VINE.bright} size={0.9} /></g>
      <g transform="translate(28, 410)"><SmallLeaf angle={-28} color={VINE.bright} /></g>

      {/* Medium leaves (simple elongated) */}
      <g transform="translate(30, 140)"><MediumLeaf angle={28} color={VINE.light} size={0.85} /></g>
      <g transform="translate(22, 310)"><MediumLeaf angle={22} color={VINE.mid} size={0.8} /></g>
      <g transform="translate(44, 455)"><MediumLeaf angle={20} color={VINE.light} size={0.75} /></g>

      {/* Medium monstera leaves */}
      <g transform="translate(8, 190)"><MediumMonsteraLeaf angle={-30} size={0.9} /></g>
      <g transform="translate(24, 280)"><MediumMonsteraLeaf angle={32} size={0.85} /></g>
      <g transform="translate(15, 370)"><MediumMonsteraLeaf angle={-10} size={0.8} /></g>

      {/* Hero monstera #1 — upper third */}
      <g transform="translate(40, 170)">
        <MonsteraLeaf size={1.1} angle={-40} />
      </g>
      {/* Hero monstera #2 — lower third */}
      <g transform="translate(26, 365)">
        <MonsteraLeaf size={0.85} angle={-20} />
      </g>

      {/* ── Flowers — animate top→bottom, mix of 4 and 8 petals ── */}
      <AnimatedFlower x={44} y={65} delay={0.1} color={FLOWER.coral} size={1.0} angle={-20} petals={4} visible={showFlowers} />
      <AnimatedFlower x={34} y={148} delay={0.28} color={FLOWER.yellow} size={1.35} angle={-12} petals={8} visible={showFlowers} />
      <AnimatedFlower x={24} y={248} delay={0.46} color={FLOWER.lavender} size={1.1} angle={8} petals={4} visible={showFlowers} />
      <AnimatedFlower x={28} y={345} delay={0.64} color={FLOWER.coral} size={1.25} angle={-5} petals={8} visible={showFlowers} />
      <AnimatedFlower x={42} y={445} delay={0.82} color={FLOWER.yellow} size={1.0} angle={-15} petals={4} visible={showFlowers} />
    </svg>
  );
}

/*
 * ══════════════════════════════════════════
 *   RIGHT  VINE  (FRONT)
 *   Asymmetric to left — different heights, angles, flower colors.
 * ══════════════════════════════════════════
 */
function RightVineFront({ showFlowers }) {
  return (
    <svg className="vine-svg" viewBox="0 0 120 500" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M68 480 Q90 435 96 375 Q100 315 98 255 Q96 195 92 150 Q86 100 76 55"
        stroke={VINE.stem} strokeWidth="4.5" strokeLinecap="round" fill="none"
      />
      <Tendril d="M96 345 Q106 336 110 322 Q112 314 108 310" />
      <Tendril d="M92 200 Q104 192 108 178 Q110 170 105 166" />
      <Tendril d="M80 80 Q90 72 94 60 Q96 54 92 50" />

      {/* Small leaves */}
      <g transform="translate(78, 68)"><SmallLeaf angle={30} color={VINE.bright} /></g>
      <g transform="translate(96, 240)"><SmallLeaf angle={18} color={VINE.bright} size={0.9} /></g>
      <g transform="translate(92, 420)"><SmallLeaf angle={25} color={VINE.bright} /></g>

      {/* Medium leaves (simple elongated) */}
      <g transform="translate(90, 135)"><MediumLeaf angle={-24} color={VINE.light} size={0.8} /></g>
      <g transform="translate(98, 320)"><MediumLeaf angle={-20} color={VINE.mid} size={0.85} /></g>
      <g transform="translate(76, 460)"><MediumLeaf angle={-22} color={VINE.light} size={0.75} /></g>

      {/* Medium monstera leaves */}
      <g transform="translate(100, 100)"><MediumMonsteraLeaf angle={140} size={0.85} /></g>
      <g transform="translate(96, 290)"><MediumMonsteraLeaf angle={-30} size={0.9} /></g>
      <g transform="translate(65, 420)"><MediumMonsteraLeaf angle={-60} size={0.8} /></g>

      {/* Hero monstera #1 — mid-height */}
      <g transform="translate(70, 180)">
        <MonsteraLeaf size={1.0} angle={120} />
      </g>
      {/* Hero monstera #2 — lower, larger */}
      <g transform="translate(94, 400)">
        <MonsteraLeaf size={1.15} angle={18} />
      </g>

      {/* ── Flowers — animate top→bottom, mix of 4 and 8 petals ── */}
      <AnimatedFlower x={78} y={72} delay={0.15} color={FLOWER.yellow} size={1.05} angle={18} petals={8} visible={showFlowers} />
      <AnimatedFlower x={90} y={160} delay={0.33} color={FLOWER.coral} size={1.3} angle={10} petals={4} visible={showFlowers} />
      <AnimatedFlower x={96} y={265} delay={0.51} color={FLOWER.blue} size={1.15} angle={-8} petals={8} visible={showFlowers} />
      <AnimatedFlower x={92} y={360} delay={0.69} color={FLOWER.lavender} size={1.2} angle={12} petals={4} visible={showFlowers} />
      <AnimatedFlower x={76} y={450} delay={0.87} color={FLOWER.coral} size={1.0} angle={5} petals={8} visible={showFlowers} />
    </svg>
  );
}

/* ══════════════════════════════════════════
   Named exports — App controls DOM order:
     VineBehind → CameraScanner → VineFront
   ══════════════════════════════════════════ */

export function VineBehind({ side }) {
  const isLeft = side === 'left';
  const ready = useSvgReady();
  return (
    <div className={`vine-layer vine-pos-${side}`}>
      <div
        className={`vine-layer-inner vine-sway vine-sway-${side}`}
        style={{ opacity: ready ? 1 : 0 }}
      >
        {isLeft ? <LeftVineBehind /> : <RightVineBehind />}
      </div>
    </div>
  );
}

export function VineFront({ side, showFlowers }) {
  const isLeft = side === 'left';
  const ready = useSvgReady();
  return (
    <div className={`vine-layer vine-pos-${side} vine-front-shadow`}>
      <div
        className={`vine-layer-inner vine-sway vine-sway-${side}`}
        style={{ opacity: ready ? 1 : 0 }}
      >
        {isLeft ? <LeftVineFront showFlowers={showFlowers} /> : <RightVineFront showFlowers={showFlowers} />}
      </div>
    </div>
  );
}
