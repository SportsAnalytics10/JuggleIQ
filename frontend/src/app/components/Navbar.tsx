import { Link } from "react-router";
import { Video, Target, Upload } from "lucide-react";
import { Button } from "./ui/button";

export function Navbar() {
  return (
    <nav className="h-16 border-b border-border bg-background">
      <div className="mx-auto max-w-[1280px] h-full px-4 flex items-center justify-between">
        {/* Logo and Navigation */}
        <div className="flex items-center gap-6">
          <Link to="/" className="flex items-center gap-2">
            <Target className="w-6 h-6 text-accent" />
            <span className="font-semibold text-lg tracking-tight">AltinhaAI</span>
          </Link>
          
          <div className="hidden md:flex items-center gap-4">
            <Link to="/upload">
              <Button variant="ghost" size="sm" className="gap-2">
                <Upload className="w-4 h-4" />
                <span>Upload</span>
              </Button>
            </Link>
            <Link to="/drill-selection">
              <Button variant="ghost" size="sm" className="gap-2">
                <Video className="w-4 h-4" />
                <span>Drills</span>
              </Button>
            </Link>
          </div>
        </div>

        {/* User Actions - Theme toggle removed */}
        <div className="flex items-center gap-2">
          {/* Future: Add user profile or settings here */}
        </div>
      </div>
    </nav>
  );
}