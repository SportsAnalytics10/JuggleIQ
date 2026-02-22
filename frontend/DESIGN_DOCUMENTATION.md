# AltinhaAI - Comprehensive Design Documentation

## 📋 Overview

AltinhaAI is a sophisticated soccer coaching application that uses computer vision and AI to analyze juggling videos. The app detects ball touches, tracks performance metrics, and provides actionable feedback to help players improve their technique.

---

## 🎨 Design System

### Color Palette

#### Light Mode
- **Background**: `#FFFFFF` (Pure white for maximum clarity)
- **Foreground**: `#252525` (Near-black for text)
- **Card**: `#F7F7F7` (Light gray for cards and panels)
- **Border**: `#E5E5E5` (Subtle borders)
- **Muted**: `#F5F5F5` (Background for secondary elements)
- **Muted Foreground**: `#737373` (Secondary text)

#### Dark Mode
- **Background**: `#252525` (Dark charcoal)
- **Foreground**: `#FAFAFA` (Off-white for text)
- **Card**: `#2E2E2E` (Slightly lighter than background)
- **Border**: `#3A3A3A` (Subtle dark borders)
- **Muted**: `#333333` (Secondary backgrounds)
- **Muted Foreground**: `#A3A3A3` (Secondary text)

#### Semantic Colors (Both Modes)
- **Primary**: `#171717` (Light) / `#7C5AE6` (Dark) - Main CTAs
- **Accent**: `#2ECC71` - Soccer field green, used for success states
- **Destructive**: `#E54D4D` - Errors and warnings

#### Chart Colors
Purpose-driven color system for soccer analytics:
- **Chart 1 - Green** (`#2ECC71`): Detected/successful touches
- **Chart 2 - Yellow** (`#F39C12`): Predicted/warning states
- **Chart 3 - Red** (`#E54D4D`): Errors/lost balls
- **Chart 4 - Blue** (`#3498DB`): Velocity/metrics
- **Chart 5 - Gray** (`#95A5A6`): Neutral data

### Typography

#### Font Families
- **Base**: `Inter` - Clean, modern sans-serif for all UI text
- **Monospace**: `Geist Mono` - Used exclusively for numeric data readouts (metrics, timestamps, scores)

#### Font Weights
- **Normal**: 400 - Body text, inputs
- **Medium**: 500 - Labels, buttons, headings
- **Semibold**: 600 - Emphasized headings
- **Bold**: 700 - Important headings, hero text

#### Scale
All text uses default sizes with Tailwind overrides when needed:
- **H1**: 2xl with medium weight
- **H2**: xl with medium weight
- **H3**: lg with medium weight
- **H4**: base with medium weight
- **Body**: base with normal weight
- **Labels/Buttons**: base with medium weight

### Spacing System
8px grid system throughout:
- **Base unit**: 8px
- **Component padding**: 16px (p-4), 24px (p-6)
- **Section spacing**: 24px (gap-6), 32px (gap-8)
- **Container max-width**: 1280px

### Border Radius
- **sm**: 6px - Small elements
- **default**: 10px - Standard components
- **md**: 12px - Cards
- **lg**: 16px - Larger containers
- **xl**: 20px - Hero elements

---

## 🧭 Navigation Structure

### Routes
```
/ (Root Layout with Navbar)
├── / (Upload Screen)
├── /drill-selection (Drill Selection)
├── /processing (Processing Screen)
├── /results (Results Dashboard)
└── /analytics (Advanced Analytics)
```

### Navbar Component
**Location**: Fixed at top, 64px height
**Features**:
- **Left Section**:
  - Logo with Target icon + "AltinhaAI" wordmark
  - Quick navigation links (Upload, Drills)
- **Right Section**:
  - Theme toggle (Light/Dark mode with Sun/Moon icons)
- **Styling**: Border bottom, background matches theme
- **Max width**: 1280px container

---

## 📱 Screen-by-Screen Design

## 1. Upload Screen (`/`)

### Layout
- Centered card design
- Maximum width: 2xl (672px)
- Vertical padding: 32px

### Header Section
- **Hero Title**: "Welcome to AltinhaAI" (4xl, bold, tight tracking)
- **Subtitle**: "AI-powered soccer juggling analysis" (xl, muted foreground)
- **Alignment**: Centered

### Upload Card
**Card styling**: Shadow-lg for depth

#### Drag & Drop Zone
- **Border**: 2px dashed, responsive to drag state
- **Padding**: 48px (p-12)
- **Background**: Muted with 50% opacity
- **States**:
  - Default: `border-border bg-muted/50`
  - Hover: `hover:border-accent/50`
  - Active drag: `border-accent bg-accent/10`

**Content Structure**:
1. **Icon Container**:
   - 64px circular div
   - Background: `bg-accent/10`
   - Upload icon (32px) in accent color
2. **Text**:
   - Primary: "Drop your video here" (lg, semibold)
   - Secondary: "or click to browse files" (sm, muted)
3. **Button**: "Choose Video" with Video icon

#### Divider
- Horizontal line with centered "or" text
- Uses Card background for text overlay

#### Record Button
- Full width
- Large size
- Camera icon + "Record Live Video" text
- Primary variant

#### Footer
- Small text (xs) with muted foreground
- "Supported formats: MP4, MOV, AVI • Max size: 500MB"

### Interactions
- Drag and drop video files
- Click to browse files
- Navigate to drill selection on file selection

---

## 2. Drill Selection Screen (`/drill-selection`)

### Layout
- Max width: 4xl (896px)
- Back button in top left corner

### Header
- **Title**: "Select Your Drill" (4xl, bold)
- **Subtitle**: "Choose the juggling technique you want to practice" (xl, muted)

### Drill Cards Grid
**Layout**: 2 columns on desktop, 1 on mobile, 24px gap

#### Card Types (4 total)

**1. Right Foot Only**
- Icon: 🦶
- Color: Green (`#2ECC71`)
- Description: "Practice juggling with your right foot exclusively"

**2. Left Foot Only**
- Icon: 🦶
- Color: Blue (`#3498DB`)
- Description: "Practice juggling with your left foot exclusively"

**3. Alternating Feet**
- Icon: ⚡
- Color: Yellow (`#F39C12`)
- Description: "Switch between left and right foot for balanced control"

**4. Freestyle**
- Icon: 🎯
- Color: Purple (`#7C5AE6`)
- Description: "Mix all techniques and get comprehensive analysis"

#### Card Design
- **Border**: 2px colored border matching drill color
- **Background**: Colored with 10% opacity, 20% on hover
- **Header**: Emoji icon (4xl) + Title + Description
- **Footer**: "Start Analysis" outline button
- **Interaction**: Entire card is clickable, navigates to processing

---

## 3. Processing Screen (`/processing`)

### Layout
- Centered vertically and horizontally
- Full viewport height minus navbar
- Max width: 2xl (672px)

### Card Design
**Shadow**: lg for prominence

#### Header Section
1. **Loading Icon**:
   - 64px circular container
   - Background: `bg-accent/10`
   - Loader2 icon (32px) with spin animation
   - Accent color
2. **Title**: "Processing Your Video" (3xl, centered)
3. **Subtitle**: "Our AI is analyzing your juggling technique" (base, centered)

#### Progress Section

**Progress Bar**:
- Height: 8px (h-2)
- Animated from 0% to 100%
- Shows current percentage below

**Processing Steps** (5 total):
1. "Uploading video" (20%)
2. "Detecting ball movement" (40%)
3. "Analyzing juggle patterns" (60%)
4. "Calculating metrics" (80%)
5. "Generating insights" (100%)

**Step Indicators**:
- **Pending**: Muted foreground with 50% opacity, gray dot
- **Active**: Accent background (10%), pulsing green dot, full opacity
- **Complete**: Muted foreground, solid green dot, green checkmark

#### Footer
- Small text: "This usually takes 10-15 seconds"
- Muted foreground

### Behavior
- Auto-advances through steps (1 second each)
- Auto-navigates to results after completion
- Real-time progress bar updates

---

## 4. Results Screen (`/results`)

### Layout
- Full width container (max 1280px)
- Two-column layout: Main content (2/3) + Sidebar (1/3)

### Top Navigation Bar
**Left**: Back button to home
**Right**: 
- "Advanced Analytics" button (primary)
- Share button (outline)
- Export button (outline)

### Page Header
- **Title**: "Analysis Results" (4xl, bold)
- **Subtitle**: "Your juggling session from [date]" (xl, muted)

### Main Content Area (Left Column)

#### Video Player Card
**Header**:
- Title: "Annotated Video"
- Description: "Watch your session with AI-detected juggles highlighted"

**Video Container**:
- Aspect ratio: 16:9
- Background: Gradient from muted to muted/50
- **Overlay Elements**:
  - Play/Pause button (80px circle, centered)
  - Detection badges:
    - Green badge: "Detected" with green dot
    - Yellow badge: "Predicted" with yellow dot
- **Progress Bar**: Bottom edge, 4px height, shows playback position

#### Charts Section
**Tabs**: "Ball Velocity" and "Peak Height"

**Chart 1 - Velocity Over Time**:
- Type: Area chart
- Color: Blue (`#3498DB`) with gradient fill
- X-axis: Time (seconds)
- Y-axis: Velocity (m/s)
- Height: 250px
- Grid: Dashed lines in `#E5E5E5`

**Chart 2 - Peak Height Over Time**:
- Type: Area chart
- Color: Green (`#2ECC71`) with gradient fill
- X-axis: Time (seconds)
- Y-axis: Height (meters)
- Height: 250px

### Sidebar (Right Column)

#### Metric Cards (6 total)
Each card follows consistent design:

**Structure**:
- Shadow: sm
- Padding: 24px (p-6)
- Value: 4xl, Geist Mono font, bold
- Label: sm, muted foreground
- Optional trend icon (up/down/neutral)

**Metrics**:
1. **Juggles Detected**: "12" (green, trend up)
2. **Current Streak**: "8" (green, trend up)
3. **Peak Height**: "1.4m" (blue, no trend)
4. **Avg Velocity**: "4.5 m/s" (blue, no trend)
5. **Lateral Drift**: "0.15m" (yellow, trend down)
6. **Accuracy Score**: "92%" (green, trend up)

#### AI Insights Card
- Background: `bg-accent/5`
- Border: `border-accent/30`
- Header: "AI Insights" (lg)

**Insights** (2-3 items):
- Emoji icon (2xl)
- Title (sm, medium weight)
- Description (xs, muted foreground)

Example insights:
- 🎯 "Great consistency!" + explanation
- ⚡ "Watch your drift" + advice

### Bottom Action
- "Analyze Another Session" button (large, centered)

---

## 5. Advanced Analytics Screen (`/analytics`)

### Layout
- Full width container (max 1280px)
- Top navigation with back button to results

### Header
- **Title**: "Advanced Analytics" (4xl, bold)
- **Subtitle**: "Deep dive into your performance with interactive visualizations" (xl, muted)

### Tab Navigation
5 tabs arranged horizontally:
1. Session Replay
2. Height Analysis
3. Footwork
4. Rhythm
5. Evolution

---

### Tab 1: Session Replay Dashboard

#### Components

**1. Video/Court Visualization**
- Aspect ratio: 16:9
- SVG-based soccer field representation
- **Elements**:
  - Center line and half-court line (dashed green)
  - Ball trail (last 10 positions, fading opacity)
  - Current ball position (8px circle)
  - Pulsing ring animation around ball
  - Color coding: Green (detected), Yellow (predicted)

**2. Overlay Metrics (Top)**
- **Left**: Time display in monospace (e.g., "15.3s")
- **Right**: Detection status badge

**3. Live Metrics (Bottom)**
Three cards with backdrop blur:
- **Velocity**: Blue color, m/s unit
- **Height**: Green color, meters unit
- **Foot**: Text display (left/right/other)

**4. Timeline**
- Height: 48px
- Background: Muted
- **Event markers**: Vertical bars (1px width)
  - Green: Detected touches
  - Yellow: Predicted touches
  - Tooltip on hover with timestamp and foot used
- **Playhead**: Red vertical line (2px) with circular handle

**5. Slider Control**
- Standard slider component
- Range: 0 to session duration
- Step: 0.1 seconds

**6. Playback Controls**
- **Left group**:
  - Skip backward (-2s)
  - Play/Pause toggle
  - Skip forward (+2s)
- **Right group**:
  - Speed selector: 0.5×, 1×, 1.5×, 2×
  - Active speed highlighted

**7. Recent Events List**
- Shows last 5 events within 1-second window
- Each row displays:
  - Timestamp (monospace)
  - Foot used (capitalize)
  - Velocity (monospace)
  - Detection status badge

---

### Tab 2: Height Consistency Heatmap

#### Summary Statistics
Three metric cards:
- **Average Height**: Monospace number with "m" unit
- **Height Range**: Difference between max and min
- **Consistency**: Percentage with green color

#### Main Heatmap
**Structure**: 5 rows × N columns

**Height Zones** (Y-axis, bottom to top):
1. "Too Low" (0-0.8m) - Red
2. "Below Target" (0.8-1.1m) - Yellow
3. "Optimal" (1.1-1.5m) - Green
4. "Above Target" (1.5-1.8m) - Yellow
5. "Too High" (1.8-3.0m) - Red

**Time Buckets** (X-axis):
- Groups of 5 touches
- Each cell: 40px height
- Color intensity: Based on consistency score
  - Higher consistency = more opaque
  - No data = 10% opacity

**Interactions**:
- Hover: Ring highlight
- Tooltip shows:
  - Touch range (e.g., "Touches 1-5")
  - Height zone label
  - Number of touches in range
  - Average consistency percentage

#### Target Line Indicator
- Dashed border icon
- Blue color (`#3498DB`)
- Label: "Target height: 1.3m"

#### Legend
Four elements in horizontal row:
- High consistency (dark color sample)
- Low consistency (light color sample)
- No data (muted sample)

#### Touch-by-Touch Grid
- Grid of 24px squares
- Color: Height zone color
- Opacity: Consistency-based
- Tooltip on hover with touch number, height, consistency

---

### Tab 3: Foot Usage Timeline

#### Summary Statistics
Four metric cards:
1. **Right Foot**: Blue, percentage + touch count
2. **Left Foot**: Green, percentage + touch count
3. **Other**: Gray, percentage + touch count (head, chest, thigh)
4. **Alternation**: Yellow, percentage of switches

#### Sub-tabs
**Timeline View** and **Distribution**

##### Timeline View Sub-tab

**1. Main Timeline**
- Height: 64px
- Background: Muted
- **Event markers**: Vertical bars (1px)
  - Blue: Right foot
  - Green: Left foot
  - Gray: Other
  - Red top bar: Unsuccessful touch
- Position based on timestamp
- Tooltip on hover

**2. Time Markers**
- 0s, middle, end time labels

**3. 5-Second Segments**
Each segment shows:
- Time range label (e.g., "0s - 5s")
- Touch count (monospace)
- Horizontal bar chart:
  - Blue section: Right foot touches
  - Green section: Left foot touches
  - Gray section: Other touches
  - Numbers shown if count > 1

##### Distribution Sub-tab

**1. Success Rate Bars**
Two progress bars:
- **Right Foot**: Blue, percentage display
- **Left Foot**: Green, percentage display
- Height: 12px, rounded full

**2. Pattern Analysis Card**
Muted background with border, displays:
- Dominant foot (Right/Left)
- Balance score (100 - difference percentage)
- Longest streak (same foot)

**3. Touch Sequence Grid**
- Grid of 32px squares
- Letter indicator: L (left), R (right), O (other)
- Background color: Foot color
- Unsuccessful: Red ring (2px)
- Tooltip with touch details

#### Legend
Four color samples:
- Blue: Right foot
- Green: Left foot
- Gray: Other
- Red ring: Unsuccessful touch

---

### Tab 4: Touch Rhythm Graph

#### Summary Statistics
Four cards:
1. **Avg Interval**: Monospace seconds
2. **Range**: Max - min interval
3. **Consistency**: Green percentage
4. **Touches/Min**: Blue, calculated as 60/avgInterval

#### Interval Distribution Chart
- Type: Bar chart
- Data: 5-touch buckets
- Height: 256px
- X-axis: Touch range labels
- Y-axis: Average interval (seconds)
- Bar color: Blue (`#3498DB`)
- Bar radius: Top corners rounded

#### Rhythm Categories

**Three categories with progress bars**:

**1. Optimal (0.8-1.2s)**
- Color: Green (`#2ECC71`)
- Shows count and percentage
- Progress bar height: 8px

**2. Fast (<0.8s)**
- Color: Yellow (`#F39C12`)
- Indicator: ⚡ (suggests rushing)
- Progress bar height: 8px

**3. Slow (>1.2s)**
- Color: Red (`#E54D4D`)
- Indicator: 🐌 (suggests too slow)
- Progress bar height: 8px

#### Rhythm Timeline
- Height: 80px
- Background: Muted with padding
- Vertical bars for each touch:
  - Height proportional to interval
  - Color: Category color
  - Min height: 4px
  - Gap: 2px between bars
- Tooltip shows touch number and interval

#### Category Distribution Over Time
- Type: Stacked bar chart
- Height: 160px
- Three layers:
  - Bottom: Optimal (green)
  - Middle: Fast (yellow)
  - Top: Slow (red)
- Legend included

#### AI Insights Card
- Background: `bg-accent/5`
- Border: `border-accent/30`
- Header: 📊 "Rhythm Analysis"

**Insight Logic**:
- Consistency ≥80%: "Excellent rhythm consistency!"
- Consistency 60-79%: "Good rhythm, room for improvement"
- Consistency <60%: "Work on maintaining consistent rhythm"
- Optimal ≥70%: "Great tempo control!"
- Optimal <70%: "Try to maintain 0.8-1.2 second rhythm"
- If fast touches dominate: "You tend to rush"
- If slow touches dominate: "Your touches are a bit slow"

#### Target Rhythm Legend
Three color samples:
- Green: Optimal (0.8-1.2s) - Best control
- Yellow: Fast (<0.8s) - May lose control
- Red: Slow (>1.2s) - Risk of dropping

---

### Tab 5: Skill Evolution Visualizer

#### Overall Stats Cards
Four gradient cards:

**1. Improvement Card**
- Gradient: Green (10% to 5%)
- Icon: TrendingUp
- Value: +N juggles (green monospace)
- Label: "juggles/session"

**2. Total Sessions Card**
- Gradient: Blue (10% to 5%)
- Icon: Target
- Value: Session count (blue monospace)
- Label: Total minutes

**3. Total Juggles Card**
- Gradient: Yellow (10% to 5%)
- Icon: Award
- Value: Total count with comma separator (yellow monospace)
- Label: "across all sessions"

**4. Skill Rating Card**
- Gradient: Purple (10% to 5%)
- Icon: 🎯 emoji
- Value: Overall skill score (colored monospace)
- Badge: Skill level (Elite/Advanced/Intermediate/Developing/Beginner)

**Skill Levels**:
- Elite: ≥90, Purple (`#7C5AE6`)
- Advanced: 75-89, Green (`#2ECC71`)
- Intermediate: 60-74, Blue (`#3498DB`)
- Developing: 40-59, Yellow (`#F39C12`)
- Beginner: <40, Gray (`#95A5A6`)

#### Sub-tabs
**Progress**, **Skill Radar**, and **Milestones**

##### Progress Sub-tab

**1. Juggles Per Session Chart**
- Type: Line chart
- Height: 256px
- Line color: Green, 3px stroke width
- Dots: 4px radius, 6px on active
- X-axis: Date labels
- Y-axis: Juggle count

**2. Performance Metrics Chart**
- Type: Area chart with gradients
- Height: 256px
- Two layers:
  - Accuracy: Green with gradient
  - Consistency: Blue with gradient
- Both with 30% opacity fills

**3. Improvement Indicators**
Three cards showing:
- Juggles: +N% with green TrendingUp icon
- Consistency: +N% with blue TrendingUp icon
- Accuracy: +N% with green TrendingUp icon

##### Skill Radar Sub-tab

**1. Radar Chart**
- Height: 320px
- Five axes:
  - Control (top)
  - Power (top-right)
  - Consistency (bottom-right)
  - Footwork (bottom-left)
  - Technique (top-left)
- Fill: Green with 30% opacity
- Stroke: Green, 2px
- Grid: Gray dashed lines

**2. Skill Breakdown Bars**
Five progress bars:
- Each shows skill name + score out of 100
- Color coding:
  - ≥80: Green
  - 60-79: Blue
  - 40-59: Yellow
  - <40: Red
- Height: 8px, rounded full

**3. Training Recommendations Card**
- Background: `bg-accent/5`
- Border: `border-accent/30`
- Icon: 💡 "Training Recommendations"
- **Logic**:
  - Control <70: "Focus on ball control drills"
  - Power <70: "Work on generating more power"
  - Consistency <70: "Practice steady rhythm and height"
  - Footwork <70: "Alternate feet more frequently"
  - Technique <70: "Focus on proper form fundamentals"
  - All ≥70: "Excellent! Try advanced drills"

##### Milestones Sub-tab

**Achievement Cards** (5 milestones):

Each card shows:
- Large emoji icon
- Achievement title
- Status text
- Checkmark if completed

**Design States**:
- **Completed**: Green background (10%), green border (30%), colored emoji, green checkmark
- **Locked**: Muted background (30%), border, grayscale emoji, no checkmark

**Milestones**:
1. 🎯 "1000 Total Juggles"
2. 🏆 "10 Training Sessions"
3. ⭐ "50 Juggles in One Session"
4. 💪 "85% Avg Accuracy"
5. ⏱️ "1 Hour Training Time"

**Next Milestone Card**
- Gradient: Purple (10% to 5%)
- Border: Purple (30%)
- Icon: 🎯
- Shows next incomplete milestone
- Progress bar (65% example)
- If all complete: "🎉 All milestones completed!"

---

## 🧩 Reusable Components

### MetricCard Component
**Props**:
- `value`: String or number
- `label`: String
- `trend`: "up" | "down" | "neutral" (optional)
- `color`: "green" | "yellow" | "red" | "blue" | "gray"

**Design**:
- Shadow: sm
- Padding: 24px
- Value: 4xl, Geist Mono, bold, leading-none, tight tracking
- Label: sm, muted foreground
- Trend icon: 20px, positioned at baseline

### Card Components (shadcn/ui)
- Base card with shadow-sm
- CardHeader with title and description
- CardContent with padding
- Consistent border-radius and backgrounds

### Button Component
**Variants**:
- **default**: Primary color, white text
- **outline**: Transparent with border
- **ghost**: Transparent, hover shows background

**Sizes**:
- **sm**: Smaller padding, icons 16px
- **default**: Standard size
- **lg**: Larger padding for CTAs

### Tabs Component
- TabsList: Grid layout matching tab count
- TabsTrigger: Active state with underline/background
- TabsContent: Animated transitions

### Progress Component
- Radix UI base
- Custom styling with accent color
- Smooth animations

### Slider Component
- Radix UI base
- Custom thumb styling
- Accent color track

### Tooltip Component
- Dark background (theme-aware)
- Border with theme border color
- 8px border radius
- Smooth fade-in animation

---

## 🎯 Design Principles

### 1. Data-First Approach
- Large, prominent metric displays using Geist Mono
- Color-coded for quick comprehension
- Contextual information hierarchy

### 2. Progressive Disclosure
- Basic metrics on results screen
- Advanced analytics in separate section
- Expandable tooltips for details

### 3. Sports-Appropriate Colors
- Green for success (soccer field association)
- Clear visual indicators for performance
- Consistent color meaning across all visualizations

### 4. Responsive Design
- Grid layouts that collapse on mobile
- Touch-friendly interactive elements
- Readable text sizes at all breakpoints

### 5. Visual Feedback
- Hover states on interactive elements
- Loading states with animations
- Pulsing indicators for real-time updates
- Smooth transitions between states

### 6. Accessibility
- Semantic HTML structure
- ARIA labels for icon-only buttons
- Sufficient color contrast ratios
- Keyboard navigation support (Radix UI)

---

## 📊 Data Visualization Strategy

### Chart Library
**Recharts** for all data visualizations:
- Area charts for continuous metrics over time
- Bar charts for categorical comparisons
- Line charts for trends
- Radar charts for multi-dimensional skills
- Stacked charts for composition

### Chart Styling
- Grid: Dashed, `#E5E5E5`
- Axes: 12px font size
- Tooltips: Match card styling
- Gradients: 5% to 95% opacity
- Animations: Smooth, not distracting

### Interactivity
- Hover tooltips with detailed info
- Click to select time ranges
- Playback controls for temporal data
- Zoom/pan capabilities where appropriate

---

## 🔄 User Flows

### Primary Flow: Video Analysis
1. Land on Upload Screen
2. Upload or record video → Navigate to Drill Selection
3. Select drill type → Navigate to Processing
4. Auto-processing (5 seconds) → Navigate to Results
5. Review metrics and annotated video
6. Optional: Explore Advanced Analytics

### Secondary Flow: Advanced Analytics
1. From Results Screen → Click "Advanced Analytics"
2. Explore 5 visualization tabs
3. Interact with replays, heatmaps, timelines
4. Review skill evolution and milestones
5. Return to Results or start new session

---

## 📱 Responsive Behavior

### Breakpoints (Tailwind)
- **sm**: 640px
- **md**: 768px
- **lg**: 1024px
- **xl**: 1280px

### Mobile Adaptations
- Single column layouts
- Stacked metric cards
- Collapsible navigation
- Touch-optimized sliders and controls
- Simplified charts with focus on key data

### Desktop Optimizations
- Multi-column layouts
- Side-by-side comparisons
- Persistent navigation
- Detailed hover states
- Full chart interactions

---

## 🎨 Animation & Motion

### Principles
- **Purposeful**: Only animate to provide feedback or guide attention
- **Fast**: 150-300ms transitions
- **Smooth**: Ease-in-out curves

### Specific Animations
1. **Loading States**:
   - Spinner rotation (continuous)
   - Pulsing dots for active steps
   - Progress bar fill (smooth)

2. **Ball Tracking**:
   - Pulsing ring (1s duration)
   - Fade opacity (1s duration)
   - Trail with decreasing opacity

3. **Hover States**:
   - Background color transitions (150ms)
   - Scale transforms on cards (200ms)
   - Opacity changes (150ms)

4. **Page Transitions**:
   - Fade in content (300ms)
   - Slide up animations for modals

---

## 🔧 Technical Implementation

### Framework Stack
- **React 18.3.1**: Component library
- **React Router 7**: Routing and navigation
- **Tailwind CSS 4**: Utility-first styling
- **Radix UI**: Accessible component primitives
- **Recharts 2**: Data visualizations
- **Lucide React**: Icon library
- **next-themes**: Dark mode support

### Component Structure
```
/src/app
├── App.tsx (Root entry)
├── routes.ts (Route definitions)
└── components/
    ├── Navbar.tsx
    ├── UploadScreen.tsx
    ├── DrillSelectionScreen.tsx
    ├── ProcessingScreen.tsx
    ├── ResultsScreen.tsx
    ├── AdvancedAnalyticsScreen.tsx
    ├── MetricCard.tsx
    ├── SessionReplayDashboard.tsx
    ├── HeightConsistencyHeatmap.tsx
    ├── FootUsageTimeline.tsx
    ├── TouchRhythmGraph.tsx
    ├── SkillEvolutionVisualizer.tsx
    └── ui/ (shadcn components)
```

### State Management
- Component-level state with `useState`
- Navigation state with React Router
- Theme state with next-themes context
- Mock data generators for prototyping

---

## 🚀 Future Enhancements

### Planned Features
1. **Real Computer Vision Integration**:
   - Connect to actual ML models
   - Real-time ball tracking
   - Pose estimation for form analysis

2. **User Accounts & History**:
   - Session history
   - Long-term progress tracking
   - Goal setting and achievements

3. **Social Features**:
   - Share results
   - Compare with friends
   - Leaderboards

4. **Advanced Drills**:
   - Guided training programs
   - Custom drill creation
   - Progressive difficulty levels

5. **Export Options**:
   - PDF reports
   - Video downloads with annotations
   - Data export (CSV/JSON)

---

## 📝 Design Decisions & Rationale

### Why These Colors?
- **Green (`#2ECC71`)**: Universal soccer field color, positive association
- **Monospace for Numbers**: Tabular alignment, professional feel
- **Muted Backgrounds**: Reduces eye strain, focuses on data

### Why This Layout?
- **Two-Column Results**: Desktop users can see video and metrics simultaneously
- **Tabbed Analytics**: Reduces cognitive load, organized exploration
- **Centered Upload**: Clear call-to-action, minimal distraction

### Why These Components?
- **shadcn/ui**: Accessible, customizable, consistent
- **Recharts**: Powerful, flexible, React-friendly
- **Radix UI**: Production-ready primitives with accessibility

---

## 🎓 Brand Identity

### Voice & Tone
- **Professional** yet approachable
- **Data-driven** but human-friendly
- **Encouraging** and constructive
- **Technical** without being intimidating

### Copy Principles
- Clear, concise labels
- Action-oriented button text
- Encouraging feedback messages
- Technical accuracy in metrics

### Visual Identity
- Clean, modern aesthetic
- Sports-focused color palette
- Data visualization as hero
- Minimal decorative elements

---

This comprehensive design system ensures AltinhaAI is not only functional but also delightful to use, helping soccer players improve their juggling skills through clear, actionable insights presented in an intuitive, visually appealing interface.
