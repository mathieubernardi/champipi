import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

#export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
#export_file_url = 'https://drive.google.com/drive/folders/1vQRTDO_NlncxSh2hzqRUjzH_Q-huXrBT'
export_file_url = 'https://drive.google.com/file/d/1QUWukVw6x7g08RTx468gwFVTIS18yFi1/view?usp=sharing'#lien du .pkl
export_file_name = 'good_model.pkl'
#export_file_url='https://drive.google.com/file/d/1tSh_AfHa89_eqIYPpH6WO6jghtFIWOKC/view?usp=sharing'#lien du .pth
#export_file_name = 'good_champi_stage-2.pth'



#classes = ['black', 'teddys', 'grizzly', 'blanc', 'pandas']
classes = ['cepes', 'girolles', 'trompettes', 'chanterelles', 'sanguins','oronges','pied_de_moutons']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
	await download_file(export_file_url, path/export_file_name)
	try:
		learn = load_learner(path, export_file_name)
		return learn
	except RuntimeError as e:
		if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
			print(e)
			message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
			raise RuntimeError(message)
		else:
			raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")