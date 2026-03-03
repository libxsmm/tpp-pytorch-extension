/****************************************************************************************
  BSD 2-Clause License

  Copyright (c) 2018, Jakub Červený
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************************/

/* The generalized hilbert functions are from: https://github.com/jakubcerveny/gilbert */

static int sgn(int x) {
  if (x < 0) { return -1; }
  if (x > 0) { return  1; }
  return 0;
}

static inline int gilbert_d2xy_r(int dst_idx, int cur_idx,
                   int *xres, int *yres,
                   int ax,int ay,
                   int bx,int by );

static inline int gilbert_xy2d_r(int cur_idx,
                   int x_dst, int y_dst,
                   int x, int y,
                   int ax, int ay,
                   int bx,int by );

static inline int gilbert_d2xy(int *x, int *y, int idx,int w,int h) {
  *x = 0;
  *y = 0;

  if (w >= h) {
    return gilbert_d2xy_r(idx,0, x,y, w,0, 0,h);
  }
  return gilbert_d2xy_r(idx,0, x,y, 0,h, w,0);
}

static inline int gilbert_d2xy_r(int dst_idx, int cur_idx,
                   int *xres, int *yres,
                   int ax,int ay,
                   int bx,int by ) {
  int nxt_idx;
  int w, h, x, y,
      dax, day,
      dbx, dby,
      di;
  int ax2, ay2, bx2, by2, w2, h2;

  w = abs(ax + ay);
  h = abs(bx + by);

  x = *xres;
  y = *yres;

  // unit major direction
  dax = sgn(ax);
  day = sgn(ay);

  // unit orthogonal direction
  dbx = sgn(bx);
  dby = sgn(by);

  di = dst_idx - cur_idx;

  if (h == 1) {
    *xres = x + dax*di;
    *yres = y + day*di;
    return 0;
  }

  if (w == 1) {
    *xres = x + dbx*di;
    *yres = y + dby*di;
    return 0;
  }

  // floor function
  ax2 = ax >> 1;
  ay2 = ay >> 1;
  bx2 = bx >> 1;
  by2 = by >> 1;

  w2 = abs(ax2 + ay2);
  h2 = abs(bx2 + by2);

  if ((2*w) > (3*h)) {
    if ((w2 & 1) && (w > 2)) {
      // prefer even steps
      ax2 += dax;
      ay2 += day;
    }

    // long case: split in two parts only
    nxt_idx = cur_idx + abs((ax2 + ay2)*(bx + by));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
      *xres = x;
      *yres = y;
      return gilbert_d2xy_r(dst_idx, cur_idx,  xres, yres, ax2, ay2, bx, by);
    }
    cur_idx = nxt_idx;

    *xres = x + ax2;
    *yres = y + ay2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax-ax2, ay-ay2, bx, by);
  }

  if ((h2 & 1) && (h > 2)) {
    // prefer even steps
    bx2 += dbx;
    by2 += dby;
  }

  // standard case: one step up, one long horizontal, one step down
  nxt_idx = cur_idx + abs((bx2 + by2)*(ax2 + ay2));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x;
    *yres = y;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, bx2,by2, ax2,ay2);
  }
  cur_idx = nxt_idx;

  nxt_idx = cur_idx + abs((ax + ay)*((bx - bx2) + (by - by2)));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x + bx2;
    *yres = y + by2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, ax,ay, bx-bx2,by-by2);
  }
  cur_idx = nxt_idx;

  *xres = x + (ax - dax) + (bx2 - dbx);
  *yres = y + (ay - day) + (by2 - dby);
  return gilbert_d2xy_r(dst_idx, cur_idx,
                        xres,yres,
                        -bx2, -by2,
                        -(ax-ax2), -(ay-ay2));
}