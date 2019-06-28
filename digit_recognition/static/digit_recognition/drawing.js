var toolkit = ['select', 'eraser', 'pencil'];

var canvas = document.getElementById("drawing_board");
var eraser = document.getElementById('eraser');
var pencil = document.getElementById('pencil');
var select = document.getElementById('select');
var collapser = document.getElementById('collapser');

var drawer = canvas.getContext('2d');
var body = document.body;

var active = false;
var tool = 'select';

drawer.fillStyle = 'red';
canvas.style.border = '0px solid darkgray';

canvas.addEventListener('click', () => toggle(canvas), {capture: true});
body.addEventListener('click', () => deactivate(canvas));

select.addEventListener('click', () => change_tool(select), {capture: true});
pencil.addEventListener('click', () => change_tool(pencil), {capture: true});
eraser.addEventListener('click', () => change_tool(eraser), {capture: true});




function toggle(object) {
    event.stopPropagation();
    if (active === false) {
        canvas.classList.replace('select', tool);
        collapser.classList.toggle('collapsed');
        $("#"+object.id).animate({borderWidth: '4px', opacity: '0.85'}, 'fast');
        active = !active;
        console.log('activated')
    } else {

    }
}

function deactivate(object) {
    if (active === true) {
        canvas.classList.replace(tool, 'select');
        active = !active;
        collapser.classList.toggle('collapsed');
        $("#"+object.id).animate({borderWidth: '0px', opacity: '0.0'}, 'fast');
        console.log('deactivated')
    }
}

function change_tool(object) {
    event.stopPropagation();
    $("#"+object.id).button('toggle');
    if (active === false) {
        toggle(canvas)
    }
    if (tool !== object.id) {
        canvas.classList.replace(tool, object.id);
        tool = object.id;
        console.log(tool)
    }
}