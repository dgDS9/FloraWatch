/**
 * Custom cartoon-style SVG icons for Florawatch.
 * Each icon uses rounded strokes, soft fills, and a playful hand-drawn feel.
 */

export function InfoIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <circle cx="12" cy="12" r="10" fill={color} opacity="0.12" />
      <circle cx="12" cy="12" r="10" stroke={color} strokeWidth="2" />
      <circle cx="12" cy="8" r="1.2" fill={color} />
      <line x1="12" y1="11" x2="12" y2="17" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  );
}

export function FlashIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path d="M13 2L4 14H12L11 22L20 10H12L13 2Z" fill={color} opacity="0.2" />
      <path d="M13 2L4 14H12L11 22L20 10H12L13 2Z" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function GalleryIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <rect x="3" y="3" width="18" height="18" rx="3" fill={color} opacity="0.12" />
      <rect x="3" y="3" width="18" height="18" rx="3" stroke={color} strokeWidth="1.8" />
      <circle cx="8.5" cy="8.5" r="2" fill={color} opacity="0.4" />
      <path d="M3 16L8 11L12 15L16 11L21 16" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function LeafIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        d="M17 8C17 8 15.5 3 9.5 3C4 3 3 8.5 3 11C3 16 7 20 12 21C12 21 12 15 12 12C12 9 14 7 17 8Z"
        fill={color}
        opacity="0.2"
      />
      <path
        d="M17 8C17 8 15.5 3 9.5 3C4 3 3 8.5 3 11C3 16 7 20 12 21C12 21 12 15 12 12C12 9 14 7 17 8Z"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M12 21C12 21 20 18 21 10C21 6 19 3 17 3"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.7"
      />
      <path
        d="M7 15C9 13 11 12 12 12"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        opacity="0.5"
      />
    </svg>
  );
}

export function SearchIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <circle cx="11" cy="11" r="7" fill={color} opacity="0.12" />
      <circle cx="11" cy="11" r="7" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      <line x1="16.5" y1="16.5" x2="21" y2="21" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
      {/* Shine mark */}
      <path d="M8 8.5C8.8 7.3 10 6.8 11.5 7" stroke="white" strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
    </svg>
  );
}

export function CheckCircleIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <circle cx="12" cy="12" r="9.5" fill={color} opacity="0.15" />
      <circle cx="12" cy="12" r="9.5" stroke={color} strokeWidth="2" />
      <path
        d="M7.5 12.5L10.5 15.5L16.5 9"
        stroke={color}
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Sparkle */}
      <circle cx="18.5" cy="5" r="1.2" fill={color} opacity="0.4" />
      <circle cx="20" cy="7" r="0.7" fill={color} opacity="0.25" />
    </svg>
  );
}

export function AlertIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        d="M12 3L2 20H22L12 3Z"
        fill={color}
        opacity="0.12"
        strokeLinejoin="round"
      />
      <path
        d="M12 3L2 20H22L12 3Z"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <line x1="12" y1="10" x2="12" y2="14" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="12" cy="17" r="1.2" fill={color} />
    </svg>
  );
}

export function CameraIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <rect x="2" y="6" width="20" height="14" rx="3" fill={color} opacity="0.13" />
      <rect x="2" y="6" width="20" height="14" rx="3" stroke={color} strokeWidth="2" strokeLinecap="round" />
      {/* Lens body */}
      <circle cx="12" cy="13" r="4" stroke={color} strokeWidth="2" />
      <circle cx="12" cy="13" r="1.5" fill={color} opacity="0.4" />
      {/* Flash bump */}
      <path d="M8.5 6V4.5C8.5 4.22 8.72 4 9 4H15C15.28 4 15.5 4.22 15.5 4.5V6" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      {/* Shine */}
      <path d="M9.5 10.5C10 9.5 11 9 12 9" stroke="white" strokeWidth="1.2" strokeLinecap="round" opacity="0.5" />
      {/* Indicator dot */}
      <circle cx="18" cy="8.5" r="1" fill={color} opacity="0.5" />
    </svg>
  );
}

export function ScannerIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      {/* Scanning rings */}
      <circle cx="12" cy="12" r="8" stroke={color} strokeWidth="1.5" opacity="0.2" />
      <circle cx="12" cy="12" r="5" stroke={color} strokeWidth="1.5" opacity="0.35" />
      <circle cx="12" cy="12" r="2" fill={color} opacity="0.5" />
      {/* Corner brackets */}
      <path d="M3 8V4.5C3 3.67 3.67 3 4.5 3H8" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      <path d="M16 3H19.5C20.33 3 21 3.67 21 4.5V8" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      <path d="M21 16V19.5C21 20.33 20.33 21 19.5 21H16" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      <path d="M8 21H4.5C3.67 21 3 20.33 3 19.5V16" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
      {/* Scan line */}
      <line x1="6" y1="12" x2="18" y2="12" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeDasharray="2 2.5" opacity="0.45" />
    </svg>
  );
}

export function HourglassIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        d="M6 2H18V7L13.5 12L18 17V22H6V17L10.5 12L6 7V2Z"
        fill={color}
        opacity="0.1"
      />
      <path
        d="M6 2H18V7L13.5 12L18 17V22H6V17L10.5 12L6 7V2Z"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Sand */}
      <path d="M9 5H15" stroke={color} strokeWidth="1.5" strokeLinecap="round" opacity="0.4" />
      <circle cx="12" cy="18" r="1" fill={color} opacity="0.5" />
      <path d="M10 19H14" stroke={color} strokeWidth="1.5" strokeLinecap="round" opacity="0.4" />
      {/* Falling grain */}
      <line x1="12" y1="12" x2="12" y2="15" stroke={color} strokeWidth="1" strokeLinecap="round" opacity="0.3" strokeDasharray="1 2" />
    </svg>
  );
}

export function SproutIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      {/* Stem */}
      <path
        d="M12 22V12"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
      />
      {/* Left leaf */}
      <path
        d="M12 15C12 15 6 14 4 9C4 9 8 6 12 10"
        fill={color}
        opacity="0.2"
      />
      <path
        d="M12 15C12 15 6 14 4 9C4 9 8 6 12 10"
        stroke={color}
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Right leaf */}
      <path
        d="M12 12C12 12 18 11 20 6C20 6 16 3 12 7"
        fill={color}
        opacity="0.2"
      />
      <path
        d="M12 12C12 12 18 11 20 6C20 6 16 3 12 7"
        stroke={color}
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Leaf veins */}
      <path d="M8 11L10.5 12.5" stroke={color} strokeWidth="1" strokeLinecap="round" opacity="0.35" />
      <path d="M16 8L13.5 9.5" stroke={color} strokeWidth="1" strokeLinecap="round" opacity="0.35" />
      {/* Sparkles */}
      <circle cx="7" cy="5" r="0.8" fill={color} opacity="0.4" />
      <circle cx="5" cy="7" r="0.5" fill={color} opacity="0.25" />
      <circle cx="19" cy="10" r="0.6" fill={color} opacity="0.3" />
    </svg>
  );
}

export function FloraLogoIcon({ size = 28, ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      {/* Background circle */}
      <circle cx="16" cy="16" r="14" fill="#3d8b3d" opacity="0.12" />
      {/* Main leaf */}
      <path
        d="M16 27C16 27 16 18 16 15C16 10 20 7 24 6C24 6 25 13 21 18C18 21.5 16 22 16 22"
        fill="#4ade80"
        opacity="0.5"
      />
      <path
        d="M16 27C16 27 16 18 16 15C16 10 20 7 24 6C24 6 25 13 21 18C18 21.5 16 22 16 22"
        stroke="#2d7a4f"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Secondary leaf */}
      <path
        d="M16 24C16 24 16 19 16 17C16 13 11 10 7 9C7 9 6 15 10 19C13 22 16 22 16 22"
        fill="#6dd67c"
        opacity="0.4"
      />
      <path
        d="M16 24C16 24 16 19 16 17C16 13 11 10 7 9C7 9 6 15 10 19C13 22 16 22 16 22"
        stroke="#3d8b3d"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Stem */}
      <path d="M16 27V14" stroke="#2d7a4f" strokeWidth="2" strokeLinecap="round" />
      {/* Leaf veins */}
      <path d="M16 16L20 12" stroke="#2d7a4f" strokeWidth="1" strokeLinecap="round" opacity="0.3" />
      <path d="M16 18L11 14" stroke="#3d8b3d" strokeWidth="1" strokeLinecap="round" opacity="0.3" />
      {/* Sparkle dots */}
      <circle cx="22" cy="4" r="1.2" fill="#fbbf24" opacity="0.7" />
      <circle cx="25" cy="7" r="0.7" fill="#fbbf24" opacity="0.4" />
      <circle cx="6" cy="7" r="0.9" fill="#fbbf24" opacity="0.5" />
    </svg>
  );
}

export function WaterDropIcon({ size = 20, color = '#4a9eed', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        d="M12 3C12 3 5 11.5 5 15.5C5 19.36 8.13 22 12 22C15.87 22 19 19.36 19 15.5C19 11.5 12 3 12 3Z"
        fill={color}
        opacity="0.2"
      />
      <path
        d="M12 3C12 3 5 11.5 5 15.5C5 19.36 8.13 22 12 22C15.87 22 19 19.36 19 15.5C19 11.5 12 3 12 3Z"
        stroke={color}
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M8.5 15C9.2 13 10.5 11.5 12 10" stroke="white" strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
      <circle cx="9" cy="17" r="0.8" fill="white" opacity="0.5" />
    </svg>
  );
}

export function CalendarIcon({ size = 20, color = '#7c6bbd', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <rect x="3" y="5" width="18" height="16" rx="3" fill={color} opacity="0.15" />
      <rect x="3" y="5" width="18" height="16" rx="3" stroke={color} strokeWidth="1.8" />
      <path d="M3 10H21" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
      <line x1="8" y1="3" x2="8" y2="7" stroke={color} strokeWidth="2" strokeLinecap="round" />
      <line x1="16" y1="3" x2="16" y2="7" stroke={color} strokeWidth="2" strokeLinecap="round" />
      <circle cx="8" cy="14.5" r="1.2" fill={color} opacity="0.5" />
      <circle cx="12" cy="14.5" r="1.2" fill={color} opacity="0.5" />
      <circle cx="16" cy="14.5" r="1.2" fill={color} opacity="0.3" />
      <circle cx="8" cy="18" r="1.2" fill={color} opacity="0.3" />
      <circle cx="12" cy="18" r="1.2" fill={color} opacity="0.3" />
    </svg>
  );
}

export function SunIcon({ size = 20, color = '#e8a317', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <circle cx="12" cy="12" r="5" fill={color} opacity="0.25" />
      <circle cx="12" cy="12" r="5" stroke={color} strokeWidth="1.8" />
      {/* Rays */}
      <line x1="12" y1="2" x2="12" y2="5" stroke={color} strokeWidth="2" strokeLinecap="round" />
      <line x1="12" y1="19" x2="12" y2="22" stroke={color} strokeWidth="2" strokeLinecap="round" />
      <line x1="2" y1="12" x2="5" y2="12" stroke={color} strokeWidth="2" strokeLinecap="round" />
      <line x1="19" y1="12" x2="22" y2="12" stroke={color} strokeWidth="2" strokeLinecap="round" />
      <line x1="4.93" y1="4.93" x2="6.76" y2="6.76" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
      <line x1="17.24" y1="17.24" x2="19.07" y2="19.07" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
      <line x1="4.93" y1="19.07" x2="6.76" y2="17.24" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
      <line x1="17.24" y1="6.76" x2="19.07" y2="4.93" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
      {/* Shine */}
      <path d="M10 10C10.8 9 11.5 8.8 12.5 9.2" stroke="white" strokeWidth="1.2" strokeLinecap="round" opacity="0.5" />
    </svg>
  );
}

export function RefreshIcon({ size = 20, color = 'currentColor', ...props }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
      <path
        d="M4 12C4 7.58 7.58 4 12 4C14.8 4 17.2 5.4 18.7 7.5"
        stroke={color}
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <path
        d="M20 12C20 16.42 16.42 20 12 20C9.2 20 6.8 18.6 5.3 16.5"
        stroke={color}
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      {/* Arrows */}
      <path d="M16 7.5H19V4.5" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M8 16.5H5V19.5" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
