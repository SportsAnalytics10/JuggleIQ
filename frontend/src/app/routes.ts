import { createBrowserRouter } from "react-router";
import { Root } from "./components/Root";
import { LandingPage } from "./components/LandingPage";
import { UploadScreen } from "./components/UploadScreen";
import { ProcessingScreen } from "./components/ProcessingScreen";
import { ResultsScreen } from "./components/ResultsScreen";
import { DrillSelectionScreen } from "./components/DrillSelectionScreen";
import { AdvancedAnalyticsScreen } from "./components/AdvancedAnalyticsScreen";
import { HomePageV2 } from "./components/HomePageV2";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: LandingPage },
      { path: "upload", Component: UploadScreen },
      { path: "drill-selection", Component: DrillSelectionScreen },
      { path: "processing", Component: ProcessingScreen },
      { path: "results", Component: ResultsScreen },
      { path: "analytics", Component: AdvancedAnalyticsScreen },
      { path: "v2", Component: HomePageV2 },
    ],
  },
]);