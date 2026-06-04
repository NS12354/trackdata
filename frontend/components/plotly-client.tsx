// Client-only Plotly component built on the lightweight dist bundle.
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);
export default Plot;
