declare module "plotly.js-dist-min";
declare module "react-plotly.js/factory" {
  import * as React from "react";
  const createPlotlyComponent: (plotly: any) => React.ComponentType<any>;
  export default createPlotlyComponent;
}
