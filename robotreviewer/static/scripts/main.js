'use strict';

require.config({
  jsx: {
    fileExtension: '.jsx'
  },
  paths: {
    'spa': "spa/scripts",

    'underscore': "spa/scripts/vendor/underscore",
    'jquery': "spa/scripts/vendor/jquery",
    'Q': 'spa/scripts/vendor/q',
    'marked': 'spa/scripts/vendor/marked',
    'backbone': 'spa/scripts/vendor/backbone',

    'react': "spa/scripts/vendor/react-dev",
    'react-dropzone': "spa/scripts/vendor/react-dropzone",

    'react-dom': "spa/scripts/vendor/react-dom",
    'immutable': "spa/scripts/vendor/immutable",

    'JSXTransformer': "spa/scripts/vendor/JSXTransformer",
  }
});


define(function (require) {
  require("app");
});
