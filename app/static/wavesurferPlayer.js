﻿    var wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: 'black',
        progressColor: 'blue'
    });

    wavesurfer.load('https://ia902606.us.archive.org/35/items/shortpoetry_047_librivox/song_cjrg_teasdale_64kb.mp3');

    var slider = document.querySelector('#slider');

    slider.oninput = function () {
        var zoomLevel = Number(slider.value);
        wavesurfer.zoom(zoomLevel);
    };