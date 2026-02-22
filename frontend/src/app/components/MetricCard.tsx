import { useState } from "react";
import { Card, CardContent } from "./ui/card";
import { ArrowUp, ArrowDown, Minus } from "lucide-react";

interface MetricCardProps {
  value: string | number;
  label: string;
  trend?: "up" | "down" | "neutral";
  color?: "green" | "yellow" | "red" | "blue" | "gray";
}

export function MetricCard({
  value,
  label,
  trend,
  color = "green",
}: MetricCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  const colorClasses = {
    green: "text-[#2ECC71]",
    yellow: "text-[#F39C12]",
    red: "text-[#E54D4D]",
    blue: "text-[#3498DB]",
    gray: "text-[#95A5A6]",
  };

  const glowColors = {
    green: "rgba(46,204,113,0.15)",
    yellow: "rgba(243,156,18,0.15)",
    red: "rgba(229,77,77,0.15)",
    blue: "rgba(52,152,219,0.15)",
    gray: "rgba(149,165,166,0.10)",
  };

  const borderGlowColors = {
    green: "rgba(46,204,113,0.4)",
    yellow: "rgba(243,156,18,0.4)",
    red: "rgba(229,77,77,0.4)",
    blue: "rgba(52,152,219,0.4)",
    gray: "rgba(149,165,166,0.3)",
  };

  const TrendIcon =
    trend === "up" ? ArrowUp : trend === "down" ? ArrowDown : Minus;

  // Determine hover transform based on trend direction
  const getHoverTransform = () => {
    if (!trend || trend === "neutral") return "translateY(0)";
    if (trend === "up") return "translateY(-8px)";
    if (trend === "down") return "translateY(8px)";
    return "translateY(0)";
  };

  const hasTrendDirection = trend === "up" || trend === "down";

  return (
    <Card
      className="shadow-sm overflow-hidden"
      style={{
        transform: isHovered && hasTrendDirection ? getHoverTransform() : "translateY(0)",
        boxShadow:
          isHovered && hasTrendDirection
            ? `0 ${trend === "up" ? "8px" : "-8px"} 24px -4px ${glowColors[color]}, 0 ${trend === "up" ? "4px" : "-4px"} 8px -2px ${glowColors[color]}`
            : "0 1px 3px rgba(0,0,0,0.1)",
        borderColor:
          isHovered && hasTrendDirection
            ? borderGlowColors[color]
            : undefined,
        transition:
          "transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease, border-color 0.3s ease",
        cursor: hasTrendDirection ? "default" : undefined,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <CardContent className="p-6 relative">
        {/* Subtle gradient overlay on hover */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              isHovered && hasTrendDirection
                ? trend === "up"
                  ? `linear-gradient(to top, transparent 0%, ${glowColors[color]} 100%)`
                  : `linear-gradient(to bottom, transparent 0%, ${glowColors[color]} 100%)`
                : "transparent",
            opacity: isHovered && hasTrendDirection ? 1 : 0,
            transition: "opacity 0.3s ease",
          }}
        />

        <div className="flex flex-col gap-2 relative z-10">
          <div className="flex items-end gap-2">
            <span
              className={`font-mono text-4xl leading-none tracking-tight ${colorClasses[color]}`}
              style={{ fontWeight: 700 }}
            >
              {value}
            </span>
            {trend && (
              <div
                style={{
                  transform:
                    isHovered && hasTrendDirection
                      ? `translateY(${trend === "up" ? "-4px" : "4px"}) scale(1.2)`
                      : "translateY(0) scale(1)",
                  transition:
                    "transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
                }}
              >
                <TrendIcon
                  className={`w-5 h-5 mb-1 ${colorClasses[color]}`}
                  style={{
                    filter:
                      isHovered && hasTrendDirection
                        ? `drop-shadow(0 0 4px ${borderGlowColors[color]})`
                        : "none",
                    transition: "filter 0.3s ease",
                  }}
                />
              </div>
            )}
          </div>
          <p className="text-sm text-muted-foreground">{label}</p>
        </div>
      </CardContent>
    </Card>
  );
}
