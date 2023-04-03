mkdir papyri/static/fa/
curl -L -o papyri/static/fa/fontawesome.css https://use.fontawesome.com/releases/v5.8.1/css/all.css

mkdir papyri/static/webfonts/
curl -L -o papyri/static/webfonts/fa-solid-900.woff2 https://use.fontawesome.com/releases/v5.8.1/webfonts/fa-solid-900.woff2
curl -L -o papyri/static/webfonts/fa-solid-900.woff https://use.fontawesome.com/releases/v5.8.1/webfonts/fa-solid-900.woff
curl -L -o papyri/static/webfonts/fa-solid-900.ttf https://use.fontawesome.com/releases/v5.8.1/webfonts/fa-solid-900.ttf


curl -L -o papyri/static/new.css https://cdn.jsdelivr.net/npm/@exampledev/new.css@1.1.2/new.min.css


mkdir -p  papyri/static/jax/input/TeX

mkdir papyri/static/extensions
curl -L -o papyri/static/mathjax.js https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js

mkdir -p papyri/static/jax
mkdir -p papyri/static/jax/extensions
mkdir -p papyri/static/jax/input
mkdir -p papyri/static/jax/input/TeX
mkdir -p papyri/static/jax/output
mkdir -p papyri/static/jax/output/HTML-CSS/
curl -L -o 'papyri/static/extensions/MathMenu.js' https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/extensions/MathMenu.js
curl -L -o 'papyri/static/extensions/MathZoom.js' https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/extensions/MathZoom.js
curl -L -o 'papyri/static/jax/input/TeX/config.js' https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/jax/input/TeX/config.js
curl -L -o 'papyri/static/jax/output/HTML-CSS/config.js' https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/jax/output/HTML-CSS/config.js
curl -L -o 'papyri/static/extensions/tex2jax.js' https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/extensions/tex2jax.js

curl -o papyri/static/myst.js https://unpkg.com/mystjs@0.0.15/dist/myst.min.js
