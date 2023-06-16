from src.hsi_moss.dataset import *
from src.hsi_moss.raster import *
import base64

# dash
from dash import Dash, html, dcc
import plotly.express as px
from dash import Dash, html, dcc, callback, Output, Input

basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
df = read_csv(basepath.joinpath("moss_copy.csv").as_posix())
df = df.loc[df['Type'] == 'sample']

dates = [
    '16 August 2018',
    '20 September 2018',
    '7 November 2018',
    '7 December 2018'
]

app = Dash(__name__)

def get_sample_text_display(sample_name):
    sample = df.loc[df['SampleId'] == sample_name]
    return f'{sample.iloc[0]["SampleId"]} | Site: {sample.iloc[0]["Site"]} | History: {sample.iloc[0]["History"]} | Species: {sample.iloc[0]["Species"]} | Treatment: {sample.iloc[0]["Treatment"]}'

app.layout = html.Div(children=[
    html.H1(children='Moss Specimen Viewer'),

    dcc.Dropdown([get_sample_text_display(u) for u in df.SampleId.unique()], get_sample_text_display('01A'), id='dropdown-selection'),
    # html.Div(id='images',children=[
    #     html.Img(),html.Img(),html.Img(),html.Img()
    # ]),
    html.Div(children=[
        html.Img(id='sample1', width='25%'),
        html.Img(id='sample2',width='25%'),
        html.Img(id='sample3',width='25%'),
        html.Img(id='sample4',width='25%'),
    ]),
    dcc.Slider(0, 100, 1,
            value=10,
            id='my-slider'
),
    html.Div(children=[
        html.Img(id='mtvi1', width='25%'),
        html.Img(id='mtvi2',width='25%'),
        html.Img(id='mtvi3',width='25%'),
        html.Img(id='mtvi4',width='25%'),
    ]),

])

@callback(
        [
    Output('sample1', 'src'),
         Output('sample2', 'src'),
         Output('sample3', 'src'),
         Output('sample4', 'src'),
           Output('mtvi1', 'src'),
         Output('mtvi2', 'src'),
         Output('mtvi3', 'src'),
         Output('mtvi4', 'src'),
        ],
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    value = value[:3]

    samples = df.loc[df['SampleId']==value]

    rgbs = []
    for idx,row in samples.iterrows():
        sample_name = f't{row["Session"]}s{row["SampleId"]}'
        specimen_mean_path = DatasetOutput(sample_name, DatasetOutputTypes.specimen_mean, basepath)
        specimen_path = specimen_mean_path.astype(DatasetOutputTypes.rgb).filepath

        with open(specimen_path.as_posix(), "rb") as image_file:
            img_data = base64.b64encode(image_file.read())
            img_data = img_data.decode()
            img_data = "{}{}".format("data:image/jpg;base64, ", img_data)
            rgbs.append(img_data)

@app.callback(
        [
         Output('mtvi1', 'src'),
         Output('mtvi2', 'src'),
         Output('mtvi3', 'src'),
         Output('mtvi4', 'src'),
        ],
    Input('my-slider', 'value'))
def update_output(value):

    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)