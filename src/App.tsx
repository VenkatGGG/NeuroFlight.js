/**
 * App.tsx - Main application component
 */
import { Suspense } from 'react';
import { Scene } from './components/Scene';
import { Dashboard } from './components/Dashboard';

function LoadingFallback() {
  return (
    <div style={styles.loading}>
      <div style={styles.spinner} />
      <p style={styles.loadingText}>Initializing Physics Engine...</p>
    </div>
  );
}

function App() {
  return (
    <div style={styles.container}>
      <Suspense fallback={<LoadingFallback />}>
        <Scene />
      </Suspense>
      <Dashboard />
      <div style={styles.footer}>
        <span>Powered by Rapier Physics + TensorFlow.js</span>
        <span style={styles.keybinds}>
          Drag to rotate · Scroll to zoom · Shift+Drag to pan
        </span>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: '100vw',
    height: '100vh',
    position: 'relative',
    overflow: 'hidden',
  },
  loading: {
    width: '100%',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(to bottom, #1a1a2e, #16213e)',
    color: 'white',
    gap: 20,
  },
  spinner: {
    width: 48,
    height: 48,
    border: '3px solid rgba(255, 255, 255, 0.1)',
    borderTopColor: '#3b82f6',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  loadingText: {
    fontSize: 14,
    color: '#94a3b8',
  },
  footer: {
    position: 'absolute',
    bottom: 20,
    left: '50%',
    transform: 'translateX(-50%)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 4,
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.4)',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  keybinds: {
    fontSize: 11,
    color: 'rgba(255, 255, 255, 0.25)',
  },
};

export default App;
