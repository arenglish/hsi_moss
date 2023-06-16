from src.hsi_moss.dataset import *
from src.hsi_moss.raster import *
import base64
from PIL import Image
import io
import numpy as np

# dash
from dash import Dash, html, dcc, State
import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input, ctx

app = Dash(__name__)


basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
df = read_csv(basepath.joinpath("moss_copy.csv").as_posix())
df = df.loc[df["Type"] == "sample"]

dates = ["16 August 2018", "20 September 2018", "7 November 2018", "7 December 2018"]


def get_sample_stats(df: DataFrame, sampleId):
    samples = df.loc[df.SampleId == sampleId]
    data = {
        "sds": samples.iloc(),
        "cr0": DataFrame(),
        "cr1": DataFrame(),
        "cr_sums": DataFrame(),
        "mtvi2": DataFrame(),
    }
    for idx, row in enumerate(samples):
        date = dates[row["Session"] - 1]
        sample_name = f't{row["Session"]}s{row["SampleId"]}'
        specimen_mean_path = DatasetOutput(
            sample_name, DatasetOutputTypes.specimen_mean, basepath
        )
        df_mean = read_csv(specimen_mean_path.filepath.as_posix())

        data["sds"] = samples[[]]
        samples.at[idx, "Name"] = sample_name
        samples.at[idx, "Date"] = date
        samples.at[idx, "Refl_Mean_Wavelengths"] = df_mean["# wavelengths"]
        samples.at[idx, "Refl_Mean"] = df_mean["reflectance"]

        # CR Data
        cr_datapath = specimen_mean_path.astype(
            DatasetOutputTypes.continuum_removal
        ).filepath.with_suffix(".csv")
        cr_data = read_csv(cr_datapath.as_posix())
        samples.at[idx, "cr0_wavelengths"] = cr_data["cr0_wavelengths"]
        samples.at[idx, "cr0_intensities"] = cr_data["cr0_intensities"]
        samples.at[idx, "cr1_wavelengths"] = cr_data["cr1_wavelengths"]
        samples.at[idx, "cr1_intensities"] = cr_data["cr1_intensities"]

    return samples


def get_sample_text_display(sample_name):
    sample = df.loc[df["SampleId"] == sample_name]
    return f'{sample.iloc[0]["SampleId"]} | Site: {sample.iloc[0]["Site"]} | History: {sample.iloc[0]["History"]} | Species: {sample.iloc[0]["Species"]} | Treatment: {sample.iloc[0]["Treatment"]}'


# df.loc[df['Type']=='sample'].groupby(['SampleId']).apply(plot_sds)

app.layout = html.Div(
    children=[
        html.H1(children="Moss Specimen Viewer"),
        dcc.Dropdown(
            [get_sample_text_display(u) for u in df.SampleId.unique()],
            get_sample_text_display("01A"),
            id="dropdown-selection",
        ),
        # html.Div(id='images',children=[
        #     html.Img(),html.Img(),html.Img(),html.Img()
        # ]),
        html.Div(
            children=[
                html.Img(id="sample1", width="25%"),
                html.Img(id="sample2", width="25%"),
                html.Img(id="sample3", width="25%"),
                html.Img(id="sample4", width="25%"),
            ]
        ),
        html.Div(
            children=[
                html.Img(id="mtvi1", width="25%"),
                html.Img(id="mtvi2", width="25%"),
                html.Img(id="mtvi3", width="25%"),
                html.Img(id="mtvi4", width="25%"),
            ]
        ),
        dcc.Slider(0, 100, 1,
            value=10,
            id='my-slider',
            marks=None),
        html.Div(id='slider-val', children='0'),
        dcc.Graph(id="graph-sds"),
        dcc.Graph(id="graph-mtvi2"),
        dcc.Graph(id="graph-crsums"),
        dcc.Graph(id="graph-cr"),
    ]
)


@callback(
    [
        Output("graph-sds", "figure"),
        Output("graph-mtvi2", "figure"),
        Output("sample1", "src"),
        Output("sample2", "src"),
        Output("sample3", "src"),
        Output("sample4", "src"),
        Output("mtvi1", "src"),
        Output("mtvi2", "src"),
        Output("mtvi3", "src"),
        Output("mtvi4", "src"),
        Output("slider-val", "children"),
    ],
    [
        Input("dropdown-selection", "value"),
        Input("my-slider", "value"),
    ],
    [
        State("graph-sds", "figure"),
        State("graph-mtvi2", "figure"),
        State("sample1", "src"),
        State("sample2", "src"),
        State("sample3", "src"),
        State("sample4", "src"),
        State("mtvi1", "src"),
        State("mtvi2", "src"),
        State("mtvi3", "src"),
        State("mtvi4", "src"),
        State('slider-val', 'children')
    ]
)
def update_graph(value, slider, state1, state2, state3, state4, state5, state6, state7, state8, state9, state10, state11):
    value = value[:3]
    samples = df.loc[df.SampleId == value]

    if ctx.triggered_id == 'my-slider':
        mtvi_val = (slider*2/100)-1
        mtvis = []
        for index, row in samples.iterrows():
            date = dates[row["Session"] - 1]
            sample_name = f't{row["Session"]}s{row["SampleId"]}'
            specimen_mean_path = DatasetOutput(
                sample_name, DatasetOutputTypes.specimen_mean, basepath
            )
            mtvi2_path = specimen_mean_path.astype(DatasetOutputTypes.mtvi2).filepath

            mtvi2_stiff = STiff(mtvi2_path.as_posix())
            alpha = np.zeros_like(mtvi2_stiff.rgb[:,:,3])
            mtvi2 = mtvi2_stiff.cube[:,:,0]
            alpha[mtvi2 > mtvi_val] = 255
            mtvi2_stiff.rgb[:,:,3] = alpha
            img = Image.fromarray(mtvi2_stiff.rgb, "RGBA")

            byteIO = io.BytesIO()
            img.save(byteIO, format="PNG")
            byteArr = byteIO.getvalue()
            mtvis.append("data:image/png;base64," + base64.b64encode(byteArr).decode())
        return state1, state2, state3, state4, state5, state6, mtvis[0], mtvis[1], mtvis[2], mtvis[3], str(mtvi_val)
    else:
        # sds
        sds_data = {"Date": [], "Wavelengths": [], "Reflectance": []}
        specimen_urls = []
        mtvi2_rgbs = []
        for index, row in samples.iterrows():
            date = dates[row["Session"] - 1]
            sample_name = f't{row["Session"]}s{row["SampleId"]}'
            specimen_mean_path = DatasetOutput(
                sample_name, DatasetOutputTypes.specimen_mean, basepath
            )
            specimen_path = specimen_mean_path.astype(DatasetOutputTypes.rgb).filepath
            with open(specimen_path.as_posix(), "rb") as image_file:
                img_data = base64.b64encode(image_file.read())
                img_data = img_data.decode()
                img_data = "{}{}".format("data:image/jpg;base64, ", img_data)
                specimen_urls.append(img_data)
            mtvi2_path = specimen_mean_path.astype(DatasetOutputTypes.mtvi2).filepath

            mtvi2_stiff = STiff(mtvi2_path.as_posix(), TiffOptions(rgb_only=True))
            img = Image.fromarray(mtvi2_stiff.rgb, "RGBA")
            byteIO = io.BytesIO()
            img.save(byteIO, format="PNG")
            byteArr = byteIO.getvalue()
            mtvi2_rgbs.append("data:image/png;base64," + base64.b64encode(byteArr).decode())

            df_mean = read_csv(specimen_mean_path.filepath.as_posix())
            sds_data["Date"] = sds_data["Date"] + [date] * len(df_mean)
            sds_data["Wavelengths"] = sds_data["Wavelengths"] + list(
                df_mean["# wavelengths"]
            )
            sds_data["Reflectance"] = sds_data["Reflectance"] + list(df_mean["reflectance"])
        sds_data = DataFrame(sds_data)

        sds_chart = px.line(sds_data, x="Wavelengths", y="Reflectance", color="Date")
        # samples = get_sample_stats(samples)
        # sd_graph = px.line(samples, x='')
        # sd_graph.add_lin
        return (
            sds_chart,
            px.line(samples, x="Session", y="cr0"),
            specimen_urls[0],
            specimen_urls[1],
            specimen_urls[2],
            specimen_urls[3],
            mtvi2_rgbs[0],
            mtvi2_rgbs[1],
            mtvi2_rgbs[2],
            mtvi2_rgbs[3],
            state11
        )


# @callback(
#     Output('images', 'children'),
#     Input('dropdown-selection', 'value')
# )
# def update_images(value):
#     samples = df.loc[df['SampleId']==value]
#     imgs = []

#     for row in samples:
#         sample_name = f't{row["Session"]}s{row["SampleId"]}'
#         stiff_path = DatasetOutput(sample_name, DatasetOutputTypes.stiff_corrected, basepath).filepath
#         rgb = STiff(stiff_path, TiffOptions(rgb_only=True)).rgb

if __name__ == "__main__":
    app.run_server(debug=True)
