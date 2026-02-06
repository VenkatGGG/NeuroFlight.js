/**
 * Dashboard.tsx - Training controls and statistics overlay
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useTrainingStore } from '../store/trainingStore';

export function Dashboard() {
  const workerRef = useRef<Worker | null>(null);
  const [episodes, setEpisodes] = useState(1000);

  const {
    isTraining,
    isPaused,
    isRunning,
    stats,
    modelWeights,
    sharedBuffer,
    setTraining,
    setPaused,
    setRunning,
    setStats,
    setObstacles,
    setModelWeights,
    setDroneState,
    initSharedBuffer,
  } = useTrainingStore();

  // Initialize shared buffer on mount
  useEffect(() => {
    initSharedBuffer();
  }, [initSharedBuffer]);

  // Initialize worker
  useEffect(() => {
    console.log('Initializing training worker...');
    workerRef.current = new Worker(new URL('../workers/TrainingWorker.ts', import.meta.url), {
      type: 'module',
    });

    workerRef.current.onerror = (e) => {
      console.error('Worker error:', e);
    };

    workerRef.current.onmessage = (event) => {
      console.log('Worker message:', event.data);
      const { type, data, error } = event.data;

      switch (type) {
        case 'progress':
          setStats({
            episode: data.episode,
            totalEpisodes: data.totalEpisodes,
            averageReward: data.averageReward,
            bestReward: data.bestReward,
            policyLoss: data.policyLoss,
            valueLoss: data.valueLoss,
            recentRewards: data.recentRewards,
          });
          break;

        case 'complete':
          setTraining(false);
          if (data.weights) {
            setModelWeights(data.weights);
          }
          setStats({
            episode: data.episode,
            totalEpisodes: data.totalEpisodes,
            averageReward: data.averageReward,
            bestReward: data.bestReward,
            recentRewards: data.recentRewards,
          });
          break;

        case 'obstacles':
          setObstacles(data.obstacles);
          break;

        case 'state':
          if (data.droneState) {
            setDroneState(data.droneState);
          }
          break;

        case 'error':
          console.error('Worker error:', error);
          setTraining(false);
          setRunning(false);
          break;
      }
    };

    return () => {
      workerRef.current?.terminate();
    };
  }, [setTraining, setRunning, setStats, setObstacles, setDroneState, setModelWeights]);

  const handleStartTraining = useCallback(() => {
    console.log('Start training clicked, worker:', workerRef.current);
    if (!workerRef.current) {
      console.error('Worker not initialized');
      return;
    }

    setTraining(true);
    setStats({ episode: 0, totalEpisodes: episodes });

    const message = {
      type: 'start',
      config: {
        totalEpisodes: episodes,
        maxStepsPerEpisode: 2000,
        stepsPerUpdate: 2048,
        visualizationMode: false,
      },
      sharedBuffer,
    };
    console.log('Sending message to worker:', message);
    workerRef.current.postMessage(message);
  }, [episodes, sharedBuffer, setTraining, setStats]);

  const handleStopTraining = useCallback(() => {
    workerRef.current?.postMessage({ type: 'stop' });
    setTraining(false);
  }, [setTraining]);

  const handlePauseResume = useCallback(() => {
    workerRef.current?.postMessage({ type: isPaused ? 'resume' : 'pause' });
    setPaused(!isPaused);
  }, [isPaused, setPaused]);

  const handleRun = useCallback(() => {
    if (!workerRef.current) return;

    if (isRunning) {
      workerRef.current.postMessage({ type: 'stop' });
      setRunning(false);
    } else {
      // Load weights first if available
      if (modelWeights) {
        workerRef.current.postMessage({
          type: 'loadWeights',
          weights: modelWeights,
        });
      }

      setRunning(true);
      workerRef.current.postMessage({
        type: 'run',
        sharedBuffer,
      });
    }
  }, [isRunning, modelWeights, sharedBuffer, setRunning]);

  const progress = stats.totalEpisodes > 0 ? (stats.episode / stats.totalEpisodes) * 100 : 0;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>NeuroFlight.js</h1>
        <p style={styles.subtitle}>Drone RL Training Environment</p>
      </div>

      <div style={styles.controls}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Episodes</label>
          <input
            type="number"
            value={episodes}
            onChange={(e) => setEpisodes(Math.max(1, parseInt(e.target.value) || 1))}
            disabled={isTraining}
            style={styles.input}
          />
        </div>

        <div style={styles.buttonGroup}>
          {!isTraining ? (
            <button onClick={handleStartTraining} style={styles.buttonPrimary}>
              Train
            </button>
          ) : (
            <>
              <button onClick={handlePauseResume} style={styles.buttonSecondary}>
                {isPaused ? 'Resume' : 'Pause'}
              </button>
              <button onClick={handleStopTraining} style={styles.buttonDanger}>
                Stop
              </button>
            </>
          )}

          <button
            onClick={handleRun}
            disabled={isTraining}
            style={{
              ...styles.buttonSuccess,
              opacity: isTraining ? 0.5 : 1,
            }}
          >
            {isRunning ? 'Stop Demo' : 'Run Demo'}
          </button>
        </div>
      </div>

      {(isTraining || stats.episode > 0) && (
        <div style={styles.stats}>
          <div style={styles.progressContainer}>
            <div style={styles.progressLabel}>
              <span>Progress</span>
              <span>
                {stats.episode} / {stats.totalEpisodes}
              </span>
            </div>
            <div style={styles.progressBar}>
              <div
                style={{
                  ...styles.progressFill,
                  width: `${progress}%`,
                }}
              />
            </div>
          </div>

          <div style={styles.statsGrid}>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Avg Reward</span>
              <span style={styles.statValue}>{stats.averageReward.toFixed(2)}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Best Reward</span>
              <span style={styles.statValue}>
                {isFinite(stats.bestReward) ? stats.bestReward.toFixed(2) : '--'}
              </span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Policy Loss</span>
              <span style={styles.statValue}>{stats.policyLoss.toFixed(4)}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Value Loss</span>
              <span style={styles.statValue}>{stats.valueLoss.toFixed(4)}</span>
            </div>
          </div>

          {stats.recentRewards.length > 0 && (
            <div style={styles.chart}>
              <RewardChart rewards={stats.recentRewards} />
            </div>
          )}
        </div>
      )}

      {modelWeights && <div style={styles.modelBadge}>Trained model ready</div>}
    </div>
  );
}

function RewardChart({ rewards }: { rewards: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rewards.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 10;

    ctx.clearRect(0, 0, width, height);

    const minReward = Math.min(...rewards);
    const maxReward = Math.max(...rewards);
    const range = maxReward - minReward || 1;

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = padding + (i / 4) * (height - 2 * padding);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw reward line
    ctx.strokeStyle = '#4ade80';
    ctx.lineWidth = 2;
    ctx.beginPath();

    rewards.forEach((reward, i) => {
      const x = padding + (i / (rewards.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((reward - minReward) / range) * (height - 2 * padding);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(74, 222, 128, 0.3)');
    gradient.addColorStop(1, 'rgba(74, 222, 128, 0)');

    ctx.fillStyle = gradient;
    ctx.lineTo(width - padding, height - padding);
    ctx.lineTo(padding, height - padding);
    ctx.closePath();
    ctx.fill();
  }, [rewards]);

  return (
    <canvas ref={canvasRef} width={280} height={100} style={{ width: '100%', height: '100px' }} />
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'absolute',
    top: 20,
    left: 20,
    width: 320,
    background: 'rgba(15, 23, 42, 0.9)',
    backdropFilter: 'blur(10px)',
    borderRadius: 16,
    padding: 20,
    color: 'white',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  header: {
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 700,
    margin: 0,
    background: 'linear-gradient(135deg, #60a5fa, #34d399)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  subtitle: {
    fontSize: 12,
    color: '#94a3b8',
    margin: '4px 0 0 0',
  },
  controls: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  inputGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  label: {
    fontSize: 13,
    color: '#94a3b8',
    width: 70,
  },
  input: {
    flex: 1,
    padding: '8px 12px',
    borderRadius: 8,
    border: '1px solid rgba(255, 255, 255, 0.1)',
    background: 'rgba(255, 255, 255, 0.05)',
    color: 'white',
    fontSize: 14,
    outline: 'none',
  },
  buttonGroup: {
    display: 'flex',
    gap: 8,
  },
  buttonPrimary: {
    flex: 1,
    padding: '10px 16px',
    borderRadius: 8,
    border: 'none',
    background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
    color: 'white',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'transform 0.1s, box-shadow 0.1s',
  },
  buttonSecondary: {
    flex: 1,
    padding: '10px 16px',
    borderRadius: 8,
    border: '1px solid rgba(255, 255, 255, 0.2)',
    background: 'rgba(255, 255, 255, 0.05)',
    color: 'white',
    fontSize: 14,
    fontWeight: 500,
    cursor: 'pointer',
  },
  buttonDanger: {
    flex: 1,
    padding: '10px 16px',
    borderRadius: 8,
    border: 'none',
    background: 'linear-gradient(135deg, #ef4444, #dc2626)',
    color: 'white',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
  },
  buttonSuccess: {
    flex: 1,
    padding: '10px 16px',
    borderRadius: 8,
    border: 'none',
    background: 'linear-gradient(135deg, #22c55e, #16a34a)',
    color: 'white',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
  },
  stats: {
    marginTop: 20,
    padding: 16,
    background: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 12,
    border: '1px solid rgba(255, 255, 255, 0.05)',
  },
  progressContainer: {
    marginBottom: 16,
  },
  progressLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 12,
    color: '#94a3b8',
    marginBottom: 6,
  },
  progressBar: {
    height: 6,
    background: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #3b82f6, #22c55e)',
    borderRadius: 3,
    transition: 'width 0.3s ease',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 12,
  },
  statItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  statLabel: {
    fontSize: 11,
    color: '#64748b',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  statValue: {
    fontSize: 18,
    fontWeight: 600,
    color: '#f1f5f9',
    fontFamily: 'monospace',
  },
  chart: {
    marginTop: 16,
    padding: '12px 0',
    borderTop: '1px solid rgba(255, 255, 255, 0.05)',
  },
  modelBadge: {
    marginTop: 16,
    padding: '8px 12px',
    background: 'rgba(34, 197, 94, 0.1)',
    border: '1px solid rgba(34, 197, 94, 0.3)',
    borderRadius: 8,
    color: '#4ade80',
    fontSize: 12,
    fontWeight: 500,
    textAlign: 'center',
  },
};
