import { motion } from 'motion/react';
import './FlowerBloom.css';

const FLOWER_COLORS = [
  '#f472b6', // pink
  '#fb923c', // orange
  '#a78bfa', // purple
  '#f9a8d4', // light pink
  '#fbbf24', // gold
  '#67e8f9', // cyan
];

/* SVG petal cluster for one flower */
function FlowerSVG({ color, size = 28 }) {
  const petalCount = 5;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {Array.from({ length: petalCount }).map((_, i) => {
        const angle = (360 / petalCount) * i - 90;
        const rad = (angle * Math.PI) / 180;
        const cx = 20 + Math.cos(rad) * 9;
        const cy = 20 + Math.sin(rad) * 9;
        return (
          <ellipse
            key={i}
            cx={cx}
            cy={cy}
            rx="7"
            ry="10"
            fill={color}
            opacity="0.9"
            transform={`rotate(${angle} ${cx} ${cy})`}
          />
        );
      })}
      {/* Center */}
      <circle cx="20" cy="20" r="5.5" fill="#fde68a" />
      <circle cx="20" cy="20" r="3" fill="#fbbf24" />
    </svg>
  );
}

export default function FlowerBloom({ show, side }) {
  if (!show) return null;

  const flowers = side === 'left'
    ? [
        { x: 6, y: '18%', delay: 0.1, color: 0, size: 30 },
        { x: 0, y: '38%', delay: 0.35, color: 1, size: 26 },
        { x: 8, y: '55%', delay: 0.55, color: 2, size: 32 },
        { x: 2, y: '72%', delay: 0.75, color: 3, size: 24 },
        { x: 10, y: '88%', delay: 0.9, color: 4, size: 28 },
      ]
    : [
        { x: -6, y: '22%', delay: 0.2, color: 2, size: 28 },
        { x: 0, y: '40%', delay: 0.4, color: 5, size: 32 },
        { x: -8, y: '58%', delay: 0.6, color: 0, size: 26 },
        { x: -2, y: '75%', delay: 0.8, color: 4, size: 30 },
        { x: -10, y: '90%', delay: 1.0, color: 1, size: 24 },
      ];

  return (
    <div className={`flower-layer flower-layer-${side}`}>
      {flowers.map((f, i) => (
        <motion.div
          key={i}
          className="flower"
          style={{ top: f.y, [side]: f.x }}
          initial={{ scale: 0, opacity: 0, rotate: -30 }}
          animate={{ scale: 1, opacity: 1, rotate: 0 }}
          transition={{
            delay: f.delay,
            type: 'spring',
            stiffness: 200,
            damping: 12,
          }}
        >
          {/* Sparkle burst behind flower */}
          <motion.div
            className="flower-sparkle"
            initial={{ scale: 0, opacity: 0.8 }}
            animate={{ scale: 2.5, opacity: 0 }}
            transition={{ delay: f.delay + 0.1, duration: 0.6 }}
          />
          <FlowerSVG color={FLOWER_COLORS[f.color]} size={f.size} />
        </motion.div>
      ))}
    </div>
  );
}
