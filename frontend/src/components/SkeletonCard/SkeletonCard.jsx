import { motion } from 'motion/react';
import './SkeletonCard.css';

export default function SkeletonCard() {
  return (
    <motion.div
      className="skeleton-card"
      initial={{ y: 40, opacity: 0, scale: 0.95 }}
      animate={{ y: 0, opacity: 1, scale: 1 }}
      exit={{ y: 30, opacity: 0, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 180, damping: 20 }}
    >
      <div className="skel-badge skel-shimmer" />
      <div className="skel-title skel-shimmer" />
      <div className="skel-subtitle skel-shimmer" />
      <div className="skel-bar skel-shimmer" />
      <div className="skel-tips">
        <div className="skel-tip skel-shimmer" />
        <div className="skel-tip skel-shimmer" />
        <div className="skel-tip skel-shimmer" />
      </div>
    </motion.div>
  );
}
