import { useMemo } from "react";
import { motion } from "motion/react";

interface SoccerBallLoaderProps {
  progress: number; // 0-100
  size?: number; // CSS size in pixels
}

// Helper: compute point on circle
function pt(
  cx: number,
  cy: number,
  angle: number,
  r: number
): [number, number] {
  return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
}

function pointsToString(points: [number, number][]): string {
  return points.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
}

export function SoccerBallLoader({
  progress,
  size = 256,
}: SoccerBallLoaderProps) {
  const cx = 100,
    cy = 100,
    ballR = 90;

  const panels = useMemo(() => {
    const innerR = 28;
    const jR = 54;
    const hexR = 80;
    const opR = 96;
    const delta = 0.28; // radians (~16°) for hexagon far vertex spread
    const opDelta = 0.35; // radians (~20°) for outer pentagon far vertex spread

    const cpAngles = Array.from(
      { length: 5 },
      (_, i) => -Math.PI / 2 + (2 * Math.PI * i) / 5
    );
    const hexAngles = cpAngles.map((a) => a + Math.PI / 5);

    const cp = cpAngles.map((a) => pt(cx, cy, a, innerR));
    const j = cpAngles.map((a) => pt(cx, cy, a, jR));
    const oA = hexAngles.map((a) => pt(cx, cy, a - delta, hexR));
    const oB = hexAngles.map((a) => pt(cx, cy, a + delta, hexR));
    const farR = cpAngles.map((a) => pt(cx, cy, a + opDelta, opR));
    const farL = cpAngles.map((a) => pt(cx, cy, a - opDelta, opR));

    const allPanels: { points: [number, number][]; type: string }[] = [];

    // Layer 0: Center pentagon (1 panel)
    allPanels.push({
      points: [cp[0], cp[1], cp[2], cp[3], cp[4]],
      type: "pentagon",
    });

    // Layer 1: 5 hexagons around center
    for (let i = 0; i < 5; i++) {
      allPanels.push({
        points: [
          cp[i],
          cp[(i + 1) % 5],
          j[(i + 1) % 5],
          oB[i],
          oA[i],
          j[i],
        ],
        type: "hexagon",
      });
    }

    // Layer 2: 5 outer pentagons
    for (let i = 0; i < 5; i++) {
      allPanels.push({
        points: [j[i], oA[i], farR[i], farL[i], oB[(i + 4) % 5]],
        type: "pentagon",
      });
    }

    return allPanels;
  }, []);

  const totalPanels = panels.length; // 11
  const fillProgress = (progress / 100) * totalPanels;
  const filledCount = Math.floor(fillProgress);
  const partialFill = fillProgress - filledCount;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      {/* Glow background */}
      <div
        className="absolute inset-0 rounded-full blur-2xl transition-opacity duration-1000"
        style={{
          background: `radial-gradient(circle, rgba(46,204,113,${progress * 0.003}) 0%, transparent 70%)`,
        }}
      />

      <svg
        viewBox="0 0 200 200"
        width={size}
        height={size}
        className="relative z-10"
      >
        <defs>
          {/* Glow filter for filled panels */}
          <filter id="panelGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Stronger glow for the active panel */}
          <filter
            id="activeGlow"
            x="-50%"
            y="-50%"
            width="200%"
            height="200%"
          >
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          <clipPath id="ballClip">
            <circle cx={cx} cy={cy} r={ballR} />
          </clipPath>

          {/* Radial gradient for ball depth */}
          <radialGradient id="ballGradient" cx="40%" cy="35%">
            <stop offset="0%" stopColor="#2a2a2a" />
            <stop offset="100%" stopColor="#151515" />
          </radialGradient>
        </defs>

        {/* Ball shadow */}
        <ellipse
          cx={cx}
          cy={200}
          rx={50}
          ry={8}
          fill="rgba(0,0,0,0.3)"
          className="blur-[2px]"
        />

        {/* Ball background */}
        <circle
          cx={cx}
          cy={cy}
          r={ballR}
          fill="url(#ballGradient)"
          stroke="#3A3A3A"
          strokeWidth="2"
        />

        {/* Panels */}
        <g clipPath="url(#ballClip)">
          {panels.map((panel, i) => {
            const isFilled = i < filledCount;
            const isFilling = i === filledCount && progress < 100;
            const isComplete = progress >= 100;

            let fillColor = "transparent";
            let fillOpacity = 0;
            let filter: string | undefined;

            if (isComplete || isFilled) {
              fillColor = "#2ECC71";
              fillOpacity = 0.9;
              filter = "url(#panelGlow)";
            } else if (isFilling) {
              fillColor = "#2ECC71";
              fillOpacity = partialFill * 0.85;
              filter = "url(#activeGlow)";
            }

            return (
              <g key={i}>
                {/* Panel fill */}
                {isFilling && !isComplete ? (
                  <motion.polygon
                    points={pointsToString(panel.points)}
                    fill={fillColor}
                    fillOpacity={fillOpacity}
                    stroke="none"
                    filter={filter}
                    animate={{
                      fillOpacity: [
                        fillOpacity * 0.6,
                        fillOpacity,
                        fillOpacity * 0.6,
                      ],
                    }}
                    transition={{
                      duration: 1.2,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  />
                ) : (
                  <polygon
                    points={pointsToString(panel.points)}
                    fill={fillColor}
                    fillOpacity={fillOpacity}
                    stroke="none"
                    filter={filter}
                    style={{
                      transition: "fill-opacity 0.4s ease-out",
                    }}
                  />
                )}

                {/* Panel outline (always visible) */}
                <polygon
                  points={pointsToString(panel.points)}
                  fill="none"
                  stroke={isFilled || isComplete ? "#25a85c" : "#3A3A3A"}
                  strokeWidth={isFilled || isComplete ? "1.2" : "1"}
                  strokeLinejoin="round"
                  style={{
                    transition: "stroke 0.3s ease",
                  }}
                />
              </g>
            );
          })}
        </g>

        {/* Ball highlight (specular) */}
        <circle
          cx={cx - 22}
          cy={cy - 25}
          r={18}
          fill="white"
          opacity={0.04}
        />
        <circle
          cx={cx - 18}
          cy={cy - 20}
          r={8}
          fill="white"
          opacity={0.06}
        />

        {/* Outer ring */}
        <circle
          cx={cx}
          cy={cy}
          r={ballR}
          fill="none"
          stroke={progress >= 100 ? "#2ECC71" : "#4A4A4A"}
          strokeWidth="2.5"
          style={{
            transition: "stroke 0.5s ease",
          }}
        />

        {/* Progress arc around the ball */}
        {progress > 0 && progress < 100 && (
          <circle
            cx={cx}
            cy={cy}
            r={ballR + 5}
            fill="none"
            stroke="#2ECC71"
            strokeWidth="2"
            strokeDasharray={`${2 * Math.PI * (ballR + 5)}`}
            strokeDashoffset={`${2 * Math.PI * (ballR + 5) * (1 - progress / 100)}`}
            strokeLinecap="round"
            opacity={0.5}
            transform={`rotate(-90 ${cx} ${cy})`}
            style={{
              transition: "stroke-dashoffset 0.8s ease-out",
            }}
          />
        )}

        {/* Completion ring */}
        {progress >= 100 && (
          <motion.circle
            cx={cx}
            cy={cy}
            r={ballR + 5}
            fill="none"
            stroke="#2ECC71"
            strokeWidth="2.5"
            strokeLinecap="round"
            initial={{ opacity: 0 }}
            animate={{ opacity: [0.4, 0.8, 0.4] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          />
        )}
      </svg>

      {/* Percentage overlay */}
      <div className="absolute inset-0 flex items-center justify-center z-20">
        <span
          className="text-[2.5rem] tabular-nums tracking-tight text-white/90"
          style={{ fontFamily: "var(--font-family-mono)" }}
        >
          {Math.round(progress)}
          <span className="text-[1.2rem] text-white/50 ml-0.5">%</span>
        </span>
      </div>
    </div>
  );
}
