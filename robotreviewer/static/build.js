({
  appDir: "./",
  baseUrl: "./scripts",
  mainConfigFile: './scripts/main.js',
  dir: "../build",
  optimize: "uglify2",
  useStrict: true,

  // call with `node r.js -o build.js`
  // add `optimize=none` to skip script optimization (useful during debugging).
  // see https://github.com/requirejs/example-multipage/
  onBuildWrite: function (moduleName, path, singleContents) {
    return singleContents.replace(/jsx!/g, '');
  },

  optimizeCss: 'standard',
  stubModules: ['jsx'],
  paths: {
    'spa': "spa/scripts",

    'underscore': "spa/scripts/vendor/underscore",
    'jquery': "spa/scripts/vendor/jquery",
    'Q': 'spa/scripts/vendor/q',
    'marked': 'spa/scripts/vendor/marked',
    'backbone': 'spa/scripts/vendor/backbone',

    'react': "spa/scripts/vendor/react-prod",
    'immutable': "spa/scripts/vendor/immutable",

    'JSXTransformer': "spa/scripts/vendor/JSXTransformer",
    'PDFJS': "spa/scripts/vendor/pdfjs/pdf"
  },

  modules: [
    {
      name: 'main',
      exclude: ["react", "JSXTransformer", "text"]
    }
  ]
})
