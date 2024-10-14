#if defined(NVIDIA)
#include "nvidia/nvidia_utility.cuh" 
#elif defined(AMD)
#include "amd/amd_utility.hpp" 
#else
    #error "You should spcify the device as NVIDIA or AMD"
#endif

#include <assert.h>
#include <unistd.h>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

/* Set the entries of host arrays to zero. */
template <typename inputtype, typename returntype>
void host_reset(inputtype *a, inputtype *b, returntype *c) {
  memset(a, 0, 16*16*sizeof(inputtype));
  memset(b, 0, 16*16*sizeof(inputtype));
  memset(c, 0, 16*16*sizeof(returntype));
}

// void printheader(FILE *outfile, const char *string) {
//   fprintf(outfile,
//           "+--------------------------------------------------------------+\n");
//   fprintf(outfile, "| %-60s |\n", string);
//   fprintf(outfile,
//           "+--------------------------------------------------------------+\n");
// }
// void printitem(FILE *outfile, const char *string) {
//   fprintf(outfile, "  | %-49s\n", string);
// }

// void printpass(FILE *outfile, bool status) {
//   if (status)
//     fprintf(outfile, " [PASS] |\n");
//   else
//     fprintf(outfile, " [FAIL] |\n");
// }
// void printfooter(FILE *outfile) {
//   fprintf(outfile,
//           "  +----------------------------------------------------------+\n\n");
// }

void printMainTest(const std::string& heading) {
    std::cout << "\n========================================================\n";
    std::cout << heading << std::endl;
    std::cout << "========================================================\n";
}

// Function to print subtitles
void printSubTest(const std::string& subtitle) {
    std::cout << "\n---------- " << subtitle << " ----------\n";
}

// Function to print regular text
void printText(const std::string& text) {
    std::cout << text << std::endl;
}

template <typename inputtype, typename returntype>
void init_host_matrices(inputtype *a, inputtype *b, returntype *c, returntype *d)
{

    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            // a[i*K_GLOBAL+j] = (inputtype)(float)(rand() % 3);
            if(j==0){
              a[i*K_GLOBAL+j] = ldexp(1., 10);  
            }
            else if(j%2==1){
              a[i*K_GLOBAL+j] = ldexp(1., -2);  
            }
            else{
              a[i*K_GLOBAL+j] = ldexp(1., -3);
            }
            // a[i*K_GLOBAL+j] = 0.0;

        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            if(j==0){
              b[i*K_GLOBAL+j] = ldexp(1., 10);  
            }
            else{
            // b[i*K_GLOBAL+j] = (inputtype)(float)(rand() % 3);
            b[i*K_GLOBAL+j]= -ldexp(1., -3);
            }
            // b[i*K_GLOBAL+j]= 0.0;
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        // c[t] =  (returntype)(rand() % 3);
        c[t] =   (returntype)(ldexp(1., 20));
        // c[t] =   (returntype)(ldexp(1., 10));        
    }
    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        d[t] =  (returntype)(0.0);
    }
}


void summerize(const std::string& GPUModel, bool sub_support,
                bool rtz, bool ru, bool rd, bool rtn,
                bool artz, bool aru, bool ard, bool artn, 
                bool one_extr, bool three_extr,bool tie_to_even, bool stick_bit,
                int min_preserve_uint,
                bool srtz, bool sru, bool srd, bool srtn) {
              std::string subnormal_support, rounding_mode, tte="N.A.", sticky="N.A.", rounding_mode_sp="N.A.";
              int extra_bit=0;
              if(sub_support){
                subnormal_support ="Yes";
              }
              else {
                subnormal_support = "No";
              }
              if(one_extr){
                extra_bit =1;
                if(three_extr){
                  extra_bit =3;
                }
              }
              if(rtz & artz){
                rounding_mode = "Trucate";
              }
              else if(rtn & artn & three_extr){
                rounding_mode = "RTN";
              }
              else{
                std::cout << "Unknown rounding mode and extra bits combination." << std::endl;
              }

              if(tie_to_even){
                tte = "Yes";
              }
              if(stick_bit){
                sticky="Yes";
              }
              if(srtz){
                rounding_mode_sp = "Truncate";
              }
              else if(srtn){
                rounding_mode_sp = "RTN";
              }
              else if(sru){
                rounding_mode_sp = "RTU";
              }
              else if(srd){
                 rounding_mode_sp = "RTD";               
              }
                    const int width = 30;
std::cout << "\n";
std::cout << std::left << std::setw(width) << "GPU"
              << std::setw(width) << "Subnormals"
              << std::setw(width) << "Extra Bits"
              << std::setw(width) << "Rounding Mode"
              << std::setw(width) << "Tie-to-Even"
              << std::setw(width) << "Sticky Bit"
              << std::setw(width) << "FMA Unit Width"
              << std::setw(width) << "Rounding Mode (Special Case)" << std::endl;

    // Print a line under the header
    std::cout << std::setfill('-') << std::setw(width * 8) << "-" << std::endl;
    std::cout << std::setfill(' '); // Reset fill character for data rows

    // Example data rows
    std::cout << std::left
              << std::setw(width) << GPUModel
              << std::setw(width) << subnormal_support
              << std::setw(width) << extra_bit
              << std::setw(width) << rounding_mode
              << std::setw(width) << tte
              << std::setw(width) << sticky
              << std::setw(width) << min_preserve_uint
              << std::setw(width) << rounding_mode_sp << std::endl;
}


