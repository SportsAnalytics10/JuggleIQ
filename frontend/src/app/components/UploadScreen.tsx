import { useState } from "react";
import { useNavigate } from "react-router";
import { Upload, Video, Camera } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";

export function UploadScreen() {
  const navigate = useNavigate();
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      // Simulate file upload and navigate to drill selection
      setTimeout(() => navigate("/drill-selection"), 500);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setTimeout(() => navigate("/drill-selection"), 500);
    }
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-[1280px]">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 tracking-tight">Welcome to AltinhaAI</h1>
          <p className="text-xl text-muted-foreground">
            AI-powered soccer juggling analysis
          </p>
        </div>

        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle>Upload or Record Video</CardTitle>
            <CardDescription>
              Upload a juggling video or record one live to analyze your technique
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Upload Zone */}
            <div
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
                dragActive
                  ? "border-accent bg-accent/10"
                  : "border-border bg-muted/50 hover:border-accent/50"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="flex flex-col items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-accent/10 flex items-center justify-center">
                  <Upload className="w-8 h-8 text-accent" />
                </div>
                <div>
                  <p className="text-lg font-semibold mb-1">
                    Drop your video here
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to browse files
                  </p>
                </div>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload">
                  <Button variant="outline" asChild>
                    <span className="cursor-pointer gap-2">
                      <Video className="w-4 h-4" />
                      Choose Video
                    </span>
                  </Button>
                </label>
              </div>
            </div>

            {/* Divider */}
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-border"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-card text-muted-foreground">or</span>
              </div>
            </div>

            {/* Record Button */}
            <Button 
              className="w-full gap-2" 
              size="lg"
              onClick={() => navigate("/drill-selection")}
            >
              <Camera className="w-5 h-5" />
              Record Live Video
            </Button>

            <div className="text-xs text-muted-foreground text-center">
              Supported formats: MP4, MOV, AVI • Max size: 500MB
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
