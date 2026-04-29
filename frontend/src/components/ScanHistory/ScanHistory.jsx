import { motion, AnimatePresence } from 'motion/react';
import usePlantImage from '../../hooks/usePlantImage';
import { useI18n } from '../../i18n/I18nContext';
import './ScanHistory.css';

function HistoryThumb({ name }) {
  const { url } = usePlantImage(name);
  if (!url) return <div className="history-thumb history-thumb--empty" />;
  return <img src={url} alt={name} className="history-thumb" />;
}

function timeAgo(ts, t) {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return t('history.justNow');
  if (mins < 60) return `${mins}${t('history.mAgo')}`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}${t('history.hAgo')}`;
  const days = Math.floor(hrs / 24);
  return `${days}${t('history.dAgo')}`;
}

export default function ScanHistory({ history, onClear }) {
  const { t, lang } = useI18n();
  const loc = (v) => (v && typeof v === 'object' ? (v[lang] || v.en) : v);
  if (!history.length) return null;

  return (
    <motion.div
      className="history-section"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4, duration: 0.4 }}
    >
      <div className="history-header">
        <span className="history-title">{t('history.title')}</span>
        <button className="history-clear" onClick={onClear}>{t('history.clear')}</button>
      </div>
      <ul className="history-list">
        <AnimatePresence>
          {history.map((item, i) => (
            <motion.li
              key={item.timestamp}
              className="history-item"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ delay: i * 0.05 }}
            >
              <HistoryThumb name={item.name} />
              <div className="history-plant">
                <span className="history-name">{item.name}</span>
                {item.commonName && <span className="history-common">{loc(item.commonName)}</span>}
              </div>
              <div className="history-meta">
                <span className="history-conf">{item.confidence}%</span>
                <span className="history-time">{timeAgo(item.timestamp, t)}</span>
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </motion.div>
  );
}
