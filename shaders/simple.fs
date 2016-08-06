# version 150

in vec2 TexCoord;
out vec4 OutColor;

uniform sampler2D tex;

void main() {
    // output color is exclusively dependent on the input texture
    // for our ray tracer, this will be the ray traced scene
    OutColor = texture(tex, TexCoord);
}
