import { motion } from 'motion/react';
import { CameraIcon, HourglassIcon } from '../Icons/Icons';
import { useI18n } from '../../i18n/I18nContext';
import './ScanButton.css';

export default function ScanButton({ onClick, disabled, scanning }) {
  const { t } = useI18n();
  return (
    <motion.button
      className="scan-button"
      onClick={onClick}
      disabled={disabled}
      whileTap={{ scale: 0.93 }}
      whileHover={{ scale: 1.04 }}
      animate={scanning ? { scale: [1, 1.04, 1] } : {}}
      transition={
        scanning
          ? { repeat: Infinity, duration: 1.2, ease: 'easeInOut' }
          : { type: 'spring', stiffness: 400, damping: 20 }
      }
    >
      <span className="scan-button-icon">
        {scanning
          ? <HourglassIcon size={22} color="#fff" />
          : <CameraIcon size={22} color="#fff" />
        }
      </span>
      <span className="scan-button-label">
        {scanning ? t('scan.scanning') : t('scan.button')}
      </span>
    </motion.button>
  );
}
