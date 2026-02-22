import { useNavigate } from "react-router";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { ArrowLeft } from "lucide-react";

const drills = [
  {
    id: "right-foot",
    title: "Right Foot Only",
    description: "Practice juggling with your right foot exclusively",
    icon: "🦶",
    color: "bg-[#2ECC71]/10 hover:bg-[#2ECC71]/20",
    borderColor: "border-[#2ECC71]/30",
  },
  {
    id: "left-foot",
    title: "Left Foot Only",
    description: "Practice juggling with your left foot exclusively",
    icon: "🦶",
    color: "bg-[#3498DB]/10 hover:bg-[#3498DB]/20",
    borderColor: "border-[#3498DB]/30",
  },
  {
    id: "alternating",
    title: "Alternating Feet",
    description: "Switch between left and right foot for balanced control",
    icon: "⚡",
    color: "bg-[#F39C12]/10 hover:bg-[#F39C12]/20",
    borderColor: "border-[#F39C12]/30",
  },
  {
    id: "freestyle",
    title: "Freestyle",
    description: "Mix all techniques and get comprehensive analysis",
    icon: "🎯",
    color: "bg-[#7C5AE6]/10 hover:bg-[#7C5AE6]/20",
    borderColor: "border-[#7C5AE6]/30",
  },
];

export function DrillSelectionScreen() {
  const navigate = useNavigate();

  const handleDrillSelect = (drillId: string) => {
    // Navigate to processing screen
    navigate("/processing");
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-[1280px]">
      <div className="mb-6">
        <Button variant="ghost" onClick={() => navigate("/")} className="gap-2">
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
      </div>

      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 tracking-tight">Select Your Drill</h1>
          <p className="text-xl text-muted-foreground">
            Choose the juggling technique you want to practice
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {drills.map((drill) => (
            <Card
              key={drill.id}
              className={`cursor-pointer transition-all shadow-sm border-2 ${drill.borderColor} ${drill.color}`}
              onClick={() => handleDrillSelect(drill.id)}
            >
              <CardHeader>
                <div className="flex items-start gap-4">
                  <div className="text-4xl">{drill.icon}</div>
                  <div className="flex-1">
                    <CardTitle className="mb-2">{drill.title}</CardTitle>
                    <CardDescription>{drill.description}</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <Button variant="outline" className="w-full">
                  Start Analysis
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
