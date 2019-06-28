var toolkit = ['select', 'erase', 'draw'];

var canvas = document.getElementById("drawing_board");
var drawer = canvas.getContext('2d');
var body = document.body;

var active = false;
var tool = 'select';

drawer.fillStyle = 'red';
canvas.style.border = '0px solid darkgray';

canvas.addEventListener('click', () => toggle(canvas), {capture: true});
body.addEventListener('click', () => deactivate(canvas));



function toggle(object) {
    event.stopPropagation();
    if (active === false) {
        $("#"+object.id).animate({borderWidth: '4px', opacity: '0.85'}, 'fast');
        active = !active;
        console.log('activated')
    } else {
        active = !active;
        $("#"+object.id).animate({borderWidth: '0px', opacity: '0.0'}, 'fast');
        console.log('deactivated')
    }
}

function deactivate(object) {
    if (active === true) {
        active = !active;
        $("#"+object.id).animate({borderWidth: '0px', opacity: '0.0'}, 'fast');
        console.log('deactivated')
    }
}