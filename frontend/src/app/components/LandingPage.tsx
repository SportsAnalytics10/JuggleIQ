import { useNavigate } from "react-router";
import { Upload, Play } from "lucide-react";
import { Button } from "./ui/button";
import { useEffect, useRef, useCallback } from "react";

export function LandingPage() {
  const navigate = useNavigate();
  const scrollRef = useRef<HTMLDivElement>(null);
  const isPausedRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        navigate("/drill-selection");
      }
    },
    [navigate]
  );

  const playerCards = [
    { name: "Marcus Silva", subtitle: "Ball tracking + trajectory" },
    { name: "Emma Rodriguez", subtitle: "Ball tracking + trajectory" },
    { name: "James Chen", subtitle: "Ball tracking + trajectory" },
    { name: "Sofia Martins", subtitle: "Ball tracking + trajectory" },
    { name: "Alex Johnson", subtitle: "Ball tracking + trajectory" },
  ];

  // Duplicate cards for seamless looping
  const duplicatedCards = [...playerCards, ...playerCards];

  // Card dimensions
  const cardWidth = 320;
  const gapWidth = 20;
  const totalOneSet = playerCards.length * (cardWidth + gapWidth);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    let animationId: number;
    let position = 0;
    const speed = 0.5; // pixels per frame

    const animate = () => {
      if (!isPausedRef.current) {
        position += speed;
        if (position >= totalOneSet) {
          position = 0;
        }
        container.style.transform = `translateX(-${position}px)`;
      }
      animationId = requestAnimationFrame(animate);
    };

    animationId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationId);
  }, [totalOneSet]);

  return (
    <div className="min-h-screen bg-[#0A0A0B] font-[Inter]">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Minimal Header */}
      <header className="absolute top-0 left-0 right-0 z-50 px-8 py-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-[#2ECC71] flex items-center justify-center">
            <div className="w-4 h-4 rounded-full border-2 border-white" />
          </div>
          <span className="text-white text-lg font-semibold tracking-tight">AltinhaAI</span>
        </div>
        
        <Button
          onClick={handleUploadClick}
          variant="ghost"
          className="text-white hover:bg-white/10 rounded-full px-6"
        >
          Get Started
        </Button>
      </header>

      {/* Section 1 - Hero Text */}
      <section className="pt-[120px] pb-16 px-4">
        <div className="max-w-[800px] mx-auto text-center">
          <div className="text-[#2ECC71] text-[11px] font-semibold uppercase tracking-[0.15em] mb-4">
            AI-POWERED JUGGLING COACH
          </div>
          <h1 className="text-white text-[56px] font-[800] leading-[1.1] tracking-[-0.03em] mb-6">
            Master the art of juggling.
          </h1>
          <p className="text-[18px] leading-[1.6] mb-16" style={{ color: 'rgba(255,255,255,0.4)' }}>
            Upload your session. Get instant feedback. Level up your touch.
          </p>
        </div>
      </section>

      {/* Section 2 - Video Bar */}
      <section className="relative mb-12">
        {/* Full-bleed video strip */}
        <div className="relative w-full h-[280px] overflow-hidden flex items-center justify-center">
          {/* Gradient overlays on edges */}
          <div 
            className="absolute left-0 top-0 bottom-0 w-[200px] z-10 pointer-events-none"
            style={{
              background: 'linear-gradient(to right, #0A0A0B 0%, rgba(10,10,11,0) 100%)'
            }}
          />
          <div 
            className="absolute right-0 top-0 bottom-0 w-[200px] z-10 pointer-events-none"
            style={{
              background: 'linear-gradient(to left, #0A0A0B 0%, rgba(10,10,11,0) 100%)'
            }}
          />
          
          {/* Single video frame */}
          <div
            className="relative w-[800px] h-[280px] rounded-[12px] overflow-hidden"
            style={{ backgroundColor: 'rgba(255,255,255,0.06)' }}
          >
            {/* Film grain texture overlay */}
            <div 
              className="absolute inset-0"
              style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.15'/%3E%3C/svg%3E")`,
                mixBlendMode: 'overlay',
              }}
            />
            
            {/* Grayscale placeholder with mock soccer content */}
            <div className="absolute inset-0 flex items-center justify-center" style={{ filter: 'grayscale(1)' }}>
              <div className="w-32 h-32 rounded-full border-4 border-white/20 flex items-center justify-center">
                <Play className="w-12 h-12 text-white/40 ml-1" />
              </div>
            </div>
          </div>
        </div>
        
        {/* Caption */}
        <div className="text-center mt-3">
          <p className="text-[12px] italic" style={{ color: 'rgba(255,255,255,0.25)' }}>
            Pelé, 1958
          </p>
        </div>
      </section>

      {/* Section 3 - Player Showcase */}
      <section className="mt-12 mb-20 px-4 md:px-8">
        <div className="mb-6">
          <p className="text-[14px] uppercase tracking-[0.1em]" style={{ color: 'rgba(255,255,255,0.4)' }}>
            See it in action
          </p>
        </div>
        
        {/* Auto-scrolling carousel - shows 3 cards */}
        <div
          className="overflow-hidden mx-auto"
          style={{ maxWidth: `${3 * cardWidth + 2 * gapWidth}px` }}
          onMouseEnter={() => { isPausedRef.current = true; }}
          onMouseLeave={() => { isPausedRef.current = false; }}
        >
          <div
            ref={scrollRef}
            className="flex"
            style={{ gap: `${gapWidth}px`, willChange: 'transform' }}
          >
            {duplicatedCards.map((player, index) => (
              <div
                key={index}
                className="flex-shrink-0 rounded-[20px] border"
                style={{
                  width: `${cardWidth}px`,
                  height: '420px',
                  backgroundColor: 'rgba(255,255,255,0.03)',
                  borderColor: 'rgba(255,255,255,0.06)',
                }}
              >
                {/* Image area with ball detection overlay */}
                <div className="relative w-full h-[340px] rounded-t-[20px] overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.05)' }}>
                  {/* Placeholder background */}
                  <div className="absolute inset-0 bg-gradient-to-br from-[#1a1a1a] to-[#0d0d0d]" />
                  
                  {/* Ball tracking visualization */}
                  <svg className="absolute inset-0 w-full h-full" viewBox="0 0 320 340">
                    {/* Trajectory line */}
                    <path
                      d="M 80 280 Q 140 180, 200 120 T 280 60"
                      stroke="#2ECC71"
                      strokeWidth="2"
                      fill="none"
                      strokeDasharray="6 4"
                      opacity="0.6"
                    />
                    
                    {/* Ball positions along trajectory */}
                    <circle cx="80" cy="280" r="8" fill="#2ECC71" opacity="0.3" />
                    <circle cx="140" cy="180" r="8" fill="#2ECC71" opacity="0.5" />
                    <circle cx="200" cy="120" r="10" fill="#2ECC71" opacity="0.7" />
                    <circle cx="260" cy="80" r="12" fill="#2ECC71" opacity="0.9" />
                    
                    {/* Current ball position with ring */}
                    <circle cx="280" cy="60" r="16" fill="#2ECC71" />
                    <circle cx="280" cy="60" r="24" fill="none" stroke="#2ECC71" strokeWidth="2" opacity="0.4" />
                  </svg>
                  
                  {/* Player silhouette suggestion */}
                  <div className="absolute bottom-8 left-8 right-8">
                    <div className="h-32 rounded-lg" style={{ backgroundColor: 'rgba(255,255,255,0.02)' }} />
                  </div>
                </div>
                
                {/* Card footer */}
                <div className="p-5">
                  <h3 className="text-white text-[16px] font-bold mb-1">
                    {player.name}
                  </h3>
                  <p className="text-[12px]" style={{ color: 'rgba(255,255,255,0.35)' }}>
                    {player.subtitle}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section 4 - Upload CTA */}
      <section className="mt-20 pb-20 px-4">
        <div className="max-w-[680px] mx-auto">
          <div
            className="relative w-full h-[320px] rounded-[24px] border flex flex-col items-center justify-center p-8"
            style={{
              backgroundColor: 'rgba(255,255,255,0.03)',
              borderWidth: '1px',
              borderStyle: 'dashed',
              borderColor: 'rgba(46,204,113,0.3)',
              boxShadow: '0 0 120px rgba(46,204,113,0.08)',
            }}
          >
            {/* Upload icon */}
            <div className="mb-6">
              <Upload className="w-12 h-12 text-[#2ECC71]" />
            </div>
            
            {/* Headline */}
            <h2 className="text-white text-[28px] font-bold mb-3 text-center">
              Try it yourself
            </h2>
            
            {/* Subtitle */}
            <p className="text-[14px] text-center mb-8 max-w-[480px]" style={{ color: 'rgba(255,255,255,0.4)' }}>
              Drop your juggling video and let AltinhaAI analyze your technique
            </p>
            
            {/* CTA Button */}
            <Button
              onClick={handleUploadClick}
              className="rounded-[12px] px-8 py-[14px] text-white font-semibold shadow-lg hover:shadow-xl transition-all hover:scale-105"
              style={{
                backgroundColor: '#2ECC71',
              }}
            >
              Upload Video
            </Button>
          </div>
        </div>
      </section>

      {/* Bottom padding for overall composition */}
      <div className="h-20" />
    </div>
  );
}