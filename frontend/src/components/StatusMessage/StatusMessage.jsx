import { motion, AnimatePresence } from 'motion/react';
import { LeafIcon, SearchIcon, CheckCircleIcon, AlertIcon } from '../Icons/Icons';
import { useI18n } from '../../i18n/I18nContext';
import './StatusMessage.css';

const STATUS_CONFIG = {
  idle: { key: 'status.idle', Icon: LeafIcon, color: '#4a7c59' },
  scanning: { key: 'status.scanning', Icon: SearchIcon, color: '#c08b30' },
  picking: { key: 'status.picking', Icon: SearchIcon, color: '#2d7a4f' },
  success: { key: 'status.success', Icon: CheckCircleIcon, color: '#2d7a4f' },
  uncertain: { key: 'status.uncertain', Icon: AlertIcon, color: '#b45309' },
  error: { key: 'status.error', Icon: AlertIcon, color: '#b94040' },
};

export default function StatusMessage({ status, isFirstScan }) {
  const { t } = useI18n();
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.idle;
  const { Icon } = config;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={status}
        className={`status-message status-${status}`}
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 8 }}
        transition={{ duration: 0.3 }}
      >
        <span className="status-icon"><Icon size={18} color={config.color} /></span>
        <span className="status-text">{t(config.key)}</span>
        {status === 'scanning' && isFirstScan && (
          <span className="status-hint">{t('status.coldStart')}</span>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
