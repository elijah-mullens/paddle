const fs = require('fs');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const html = mathjax.document('', {InputJax: new TeX({packages: AllPackages}), OutputJax: new SVG({fontCache: 'none'})});
const results = JSON.parse(fs.readFileSync(0, 'utf8')).map(({tex, display}) => {
 const node = html.convert(tex, {display, em: 16, ex: 8, containerWidth: 700});
 const svg = adaptor.outerHTML(adaptor.firstChild(node));
 if (svg.includes('data-mjx-error')) throw new Error('Equation failed: '+tex);
 return svg;
});
process.stdout.write(JSON.stringify(results));
