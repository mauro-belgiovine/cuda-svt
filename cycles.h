#pragma once

/*
  This file is part of the apelink user library.
  Copyright (C) 2012 INFN.

  This library is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  CVS $Id: cycles.h,v 1.6 2012/05/22 14:44:48 rossetti Exp $

*/

typedef unsigned long long cycles_t;
typedef unsigned long long us_t;
typedef unsigned long long ns_t;

#ifdef USE_GCC_RDTSC_INTRINSICS

#include <x86intrin.h>
#define rdtscll(val)                                            \
        ((val) = __rdtsc())

#else //USE_CUSTOM_DEF_OF_RDTSC

//#define CONFIG_X86_64

#if defined(CONFIG_X86_64) || defined(__x86_64__)
#define DECLARE_ARGS(val, low, high)    unsigned low, high
#define EAX_EDX_VAL(val, low, high)     ((low) | ((unsigned long long)(high) << 32))
#define EAX_EDX_ARGS(val, low, high)    "a" (low), "d" (high)
#define EAX_EDX_RET(val, low, high)     "=a" (low), "=d" (high)
#else
#warning "using 32bit TSC asm inline"
#define DECLARE_ARGS(val, low, high)    unsigned long long val
#define EAX_EDX_VAL(val, low, high)     (val)
#define EAX_EDX_ARGS(val, low, high)    "A" (val)
#define EAX_EDX_RET(val, low, high)     "=A" (val)
#endif

static __always_inline unsigned long long __native_read_tsc(void)
{
        DECLARE_ARGS(val, low, high);

        asm volatile("rdtsc" : EAX_EDX_RET(val, low, high));

        return EAX_EDX_VAL(val, low, high);
}
#define rdtscll(val)                                            \
        ((val) = __native_read_tsc())

#endif //USE_CUSTOM_DEF_OF_RDTSC

#if 0
// this implementation is BROKEN on x86_64 !!!

/*
 * both i386 and x86_64 returns 64-bit value in edx:eax, but gcc's "A"
 * constraint has different meanings. For i386, "A" means exactly
 * edx:eax, while for x86_64 it doesn't mean rdx:rax or edx:eax. Instead,
 * it means rax *or* rdx.
 */

#define rdtscll(val)                                    \
        __asm__ __volatile__ ("rdtsc" : "=A" (val))

#endif


extern cycles_t cycles2us;
extern cycles_t cycles2ns;

static inline cycles_t get_cycles(void)
{
	unsigned long long ret = 0;
	rdtscll(ret);
	return ret;
}

#define timeval_sub_us(T2, T1) ((cycles_t)(T2.tv_sec-T1.tv_sec)*1000000 + (cycles_t)(T2.tv_usec-T1.tv_usec))

void calibrate_cycles(void);

static inline us_t cycles_to_us(cycles_t c)
{
        static int cycles_calibrated = 0;
        if(!cycles_calibrated) {
                cycles_calibrated = 1;
                calibrate_cycles();
        }
        return c / cycles2us;
}

static inline ns_t cycles_to_ns(cycles_t c)
{
        static int cycles_calibrated = 0;
        if(!cycles_calibrated) {
                cycles_calibrated = 1;
                calibrate_cycles();
        }
        return (c * 1000 / cycles2us);
}


/*
 * Local variables:
 *  c-indent-level: 8
 *  c-basic-offset: 8
 *  tab-width: 8
 *  indent-tabs-mode: nil
 * End:
 */

