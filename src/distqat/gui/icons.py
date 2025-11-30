
ICON_SVG = '''
        <svg
        width="512"
        height="512"
        viewBox="0 0 512 512"
        xmlns="http://www.w3.org/2000/svg"
        >
        <!-- Background -->
        <defs>
            <!-- Dark blue gradient background -->
            <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#071427"/>
            <stop offset="100%" stop-color="#04101F"/>
            </linearGradient>

            <!-- Main metal gradient for the A -->
            <linearGradient id="metalGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#F7FCFF"/>
            <stop offset="40%" stop-color="#D9E8FF"/>
            <stop offset="70%" stop-color="#B0C7E5"/>
            <stop offset="100%" stop-color="#7FA1D8"/>
            </linearGradient>

            <!-- Edge highlight gradient -->
            <linearGradient id="edgeGlow" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#39AFFF"/>
            <stop offset="100%" stop-color="#8EE6FF"/>
            </linearGradient>

            <!-- Subtle circuit stroke gradient -->
            <linearGradient id="circuitGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#1E6FFF" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="#1E6FFF" stop-opacity="0.1"/>
            </linearGradient>

            <!-- Glow around the A -->
            <filter id="glow">
            <feGaussianBlur stdDeviation="6" result="blur"/>
            <feColorMatrix
                in="blur"
                type="matrix"
                values="0 0 0 0 0.24
                        0 0 0 0 0.72
                        0 0 0 0 1
                        0 0 0 0.8 0"
            />
            </filter>
        </defs>

        <!-- Rounded square background -->
        <rect
            x="32"
            y="32"
            width="448"
            height="448"
            rx="96"
            ry="96"
            fill="url(#bgGrad)"
        />

        <!-- Subtle inner border -->
        <rect
            x="40"
            y="40"
            width="432"
            height="432"
            rx="88"
            ry="88"
            fill="none"
            stroke="#1D4F8F"
            stroke-width="3"
        />

        <!-- Circuit traces (simplified decorative lines) -->
        <g stroke="url(#circuitGrad)" stroke-width="2" fill="none" opacity="0.6">
            <!-- Left side traces -->
            <path d="M120 130 L120 210 L145 240" />
            <path d="M140 150 L140 220 L170 250" />
            <!-- Right side traces -->
            <path d="M392 140 L392 220 L365 245" />
            <path d="M370 165 L370 230 L340 260" />
            <!-- Bottom traces -->
            <path d="M180 400 L220 380 L260 390 L300 370" />
            <path d="M210 420 L260 405 L310 415" />
        </g>

        <!-- Glow behind the A -->
        <g filter="url(#glow)" opacity="0.6">
            <path
            d="M256 108
                L140 384
                L188 384
                L256 228
                L324 384
                L372 384
                Z"
            fill="#1E6FFF"
            />
            <path
            d="M132 332
                C150 270, 190 230, 240 214
                C300 194, 360 215, 390 260
                C368 248, 326 244, 290 252
                C246 262, 208 286, 188 316
                C176 334, 170 352, 168 368
                L132 368 Z"
            fill="#1E6FFF"
            />
        </g>

        <!-- Main stylized A -->
        <g>
            <!-- Outer A shape -->
            <path
            d="M256 104
                L132 392
                L188 392
                L256 230
                L324 392
                L380 392
                Z"
            fill="url(#metalGrad)"
            />

            <!-- Inner cut-out to make the A hollow -->
            <path
            d="M256 150
                L170 352
                L201 352
                L256 228
                L311 352
                L342 352
                Z"
            fill="#06101E"
            />

            <!-- Lower curved stroke (echo of original logo) -->
            <path
            d="M134 344
                C151 275, 197 235, 246 219
                C295 203, 352 216, 384 246
                C346 238, 312 240, 278 248
                C238 258, 208 276, 190 300
                C176 319, 170 336, 166 356
                L134 356 Z"
            fill="url(#metalGrad)"
            />

            <!-- Inner dark of lower curve -->
            <path
            d="M156 346
                C166 304, 198 276, 236 262
                C270 249, 310 246, 340 250
                C311 248, 283 251, 255 260
                C225 269, 204 283, 190 302
                C182 312, 176 326, 173 338
                L156 338 Z"
            fill="#040A14"
            />

            <!-- Thin edge highlight on right leg -->
            <path
            d="M324 392 L380 392 L256 104 L256 136 Z"
            fill="url(#edgeGlow)"
            opacity="0.7"
            />

            <!-- Small circular "sensor" at the junction of A -->
            <circle cx="268" cy="246" r="12" fill="#06101E" />
            <circle cx="268" cy="246" r="7" fill="url(#edgeGlow)" />
        </g>
        </svg>
        '''

